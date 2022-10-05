import logging
import numpy as np
import scipy.spatial
import matplotlib
import matplotlib.pyplot as plt
import rasterio
import time

from matplotlib import _png

from os import path
from rasterio.windows import Window
from itertools import combinations
from matplotlib.cm import cool

from uncoverml import mpiops
from uncoverml.shapley import select_subplot_grid_dims

log = logging.getLogger(__name__)

"""
Never use more than this many x's to compute a distance matrix
(save memory!)
"""
distance_partition_size = 10000


def sum_axis_0(x, y, dtype):
    """
    Reduce operation that sums 2 arrays on axis zero
    """
    s = np.sum(np.vstack((x, y)), axis=0)
    return s

"""
MPI reduce operation for summing over axis 0
"""
sum0_op = mpiops.MPI.Op.Create(sum_axis_0, commute=True)


class TrainingData:
    """
    Light wrapper for the indices and values of training data

    Parameters
    ----------
    indices : ndarray
        length N array of the indices of the input data that have classes
        assigned

    classes : ndarray
        length N int array of the class values at locations specified by
        indices
    """
    def __init__(self, indices, classes):
        self.indices = indices
        self.classes = classes


class KMeans:
    """
    Model object implementing learn and predict with K-means

    Parameters
    ----------
    k : int > 0
        The number of classes to cluster the data into

    oversample_factor: int > 1
        Controls the number of samples draws as part of [1] in the
        initialisation step. More mpi nodes will increase the total number
        of points. Consider values of 1 for more than about 16 nodes

    References
    ----------
    .. [1] Bahmani, Bahman, Benjamin Moseley, Andrea
    Vattani, Ravi Kumar, and Sergei Vassilvitskii. "Scalable k-means++."
    Proceedings of the VLDB Endowment 5, no. 7 (2012): 622-633.
    """
    def __init__(self, k, oversample_factor):
        self.k = k
        self.oversample_factor = oversample_factor

    def learn(self, x, indices=None, classes=None):
        """
        Find the cluster centres using k-means||

        Parameters
        ----------
        x : ndarray
            (n_samples, n_dimensions) length array containing the training
            samples to cluster

        indices : ndarray
            (n_samples) length integer array giving the locations in `x`
            where labels exist

        classes : ndarray
            (n_samples) length integer array giving the class assignments
            of points in x in locations given by `indices`

        """
        if indices is not None and classes is not None:
            log.info("Class labels found. Using semi-supervised k-means")
            training_data = TrainingData(indices, classes)
        else:
            log.info("No class labels found. Using unsupervised k-means")
            training_data = None
        C_init = initialise_centres(x, self.k, self.oversample_factor,
                                    training_data)
        log.info("Initialising full K-means with k-means|| output")
        C_final, _ = run_kmeans(x, C_init, self.k,
                                training_data=training_data)
        self.centres = C_final

    def predict(self, x, *args, **kwargs):
        y_star, _ = compute_class(x, self.centres)
        # y_star = y_star[:, np.newaxis].astype(float)
        y_star = y_star.astype(float)
        return y_star

    def get_predict_tags(self):
        tags = ['class']
        return tags


def kmean_distance2(x, C):
    """Compute squared euclidian distance to the nearest cluster centre

    Parameters
    ----------
    x : ndarray
        (n, d) array of n d-dimensional points
    C : ndarray
        (k, d) array of k cluster centres

    Returns
    -------
    d2_x : ndarray
        (n,) length array of distances from each x to the nearest centre
    """
    # To save memory we partition the computation
    nsplits = max(1, int(x.shape[0]/distance_partition_size))
    splits = np.array_split(x, nsplits)
    d2_x = np.empty(x.shape[0])
    idx = 0
    for x_i in splits:
        n_i = x_i.shape[0]
        D2_x = scipy.spatial.distance.cdist(x_i, C, metric='sqeuclidean')
        d2_x[idx:idx + n_i] = np.amin(D2_x, axis=1)
        idx += n_i
    return d2_x


def compute_weights(x, C):
    """Number of points in x assigned to each centre c in C

    Parameters
    ----------
    x : ndarray
        (n, d) array of n d-dimensional points
    C : ndarray
        (k, d) array of k cluster centres

    Returns
    -------
    weights : ndarray
        (k,) length array giving number of x closest to each c in C
    """
    nsplits = max(1, int(x.shape[0]/distance_partition_size))
    splits = np.array_split(x, nsplits)
    closests = np.empty(x.shape[0], dtype=int)
    idx = 0
    for x_i in splits:
        n_i = x_i.shape[0]
        D2_x = scipy.spatial.distance.cdist(x_i, C, metric='sqeuclidean')
        closests[idx: idx+n_i] = np.argmin(D2_x, axis=1)
        idx += n_i
    weights = np.bincount(closests, minlength=C.shape[0])
    return weights


def weighted_starting_candidates(X, k, l):
    """Generate (weighted) candidates to initialise the full k-means

    See the kmeans|| algorithm/paper for details. The goal is to find
    points that are good starting cluster centres for a full kmeans using
    only log(n) passes through the data

    Parameters
    ----------
    X : ndarray
        (n, d) array of n d-dimensional points to be clustered
    k : int > 0
        number of clusters
    l : float > 0
        The 'oversample factor' that controls how many candidates are found.
        Candidates are found independently on each node so this can be smaller
        with a bigger computation.

    Returns
    -------
    w : ndarray
        The 'weights' of the cluster centres, which are the number of points
        in X closest to each centre
    C : ndarray
        The cluster centres themselves. The total candidates is not known
        beforehand so the array will be shaped (z, d) where z is some number
        that increases with l.
    """
    # sample uniformly 1 point from X
    C = None
    if mpiops.chunk_index == 0:
        idx = np.random.choice(X.shape[0])
        C = [X[idx]]
    C = mpiops.comm.bcast(C, root=0)
    d2_x = kmean_distance2(X, C)
    # Figure out how many iterations to do. Roughly log n.
    phi_x_c_local = np.sum(d2_x)
    phi_x_c = mpiops.comm.allreduce(phi_x_c_local, op=mpiops.MPI.SUM)
    psi = int(round(np.log(phi_x_c)))
    log.info("kmeans|| using {} sampling iterations".format(psi))
    for i in range(psi):
        d2_x = kmean_distance2(X, C)
        phi_x_c_local = np.sum(d2_x)
        # note the oversample factor increasing the probabilities of
        # points being drawn
        probs = (l*d2_x/phi_x_c_local if phi_x_c_local > 0
                 else np.ones(d2_x.shape[0]) / float(d2_x.shape[0]))
        draws = np.random.rand(probs.shape[0])
        hits = draws <= probs
        new_c = X[hits]
        C = np.concatenate([C] + mpiops.comm.allgather(new_c), axis=0)
        log.info("it {}\tcandidates: {}".format(i, C.shape[0]))

    w = compute_weights(X, C)
    return w, C


def compute_class(X, C, training_data=None):
    """
    Find the closest cluster centre for each x in X

    This returns which cluster centre each X belongs to, with optional
    semi-supervised training data that will force an assignment of a point
    to a particular class

    Parameters
    ----------
    X : ndarray
        (n, d) array of n d-dimensional points to be evaluated
    C : ndarray
        (k, d) array of cluster centres, associated with classes 0..k-1
    training_data : TrainingData (optional)
        instance of TrainingData containing fixed class assignments for
        particular points

    Returns
    -------
    classes : ndarray
        (n,) int array of class assignments (0..k-1) for each x in X
    cost : float
        The total 'cost' of the assignment, which is the average
        distance of all points to their assigned centre
    """
    # we split up X into partitions to use memory more effectively
    nsplits = max(1, int(X.shape[0]/distance_partition_size))
    splits = np.array_split(X, nsplits)
    classes = np.empty(X.shape[0], dtype=int)
    idx = 0
    local_cost = 0
    for x_i in splits:
        n_i = x_i.shape[0]
        D2_x = scipy.spatial.distance.cdist(x_i, C, metric='sqeuclidean')
        classes_i = np.argmin(D2_x, axis=1)
        classes[idx:idx+n_i] = classes_i
        x_indices = np.arange(classes_i.shape[0])
        local_cost += np.mean(D2_x[x_indices, classes_i])
        idx += n_i
    x_indices = np.arange(classes.shape[0])

    cost = mpiops.comm.allreduce(local_cost)
    # force assignment of the training data
    if training_data:
        classes[training_data.indices] = training_data.classes
    return classes, cost


def centroid(X, weights=None):
    """Compute the centroid of a set of points X

    The points X may have repetitions given by the weights.

    Parameters
    ----------
    X : ndarray
        (n, d) array of n d-dimensional points
    weights : ndarray (optional)
        (n,) array of weights giving the repetition (or mass?) of each X

    Returns
    -------
    centroid : ndarray
        (d,) length array, the d-dimensional centroid point of all x in X.
    """
    centroid = np.zeros(X.shape[1])
    if weights is not None:
        local_count = np.sum(weights)
        local_sum = np.sum(X * weights, axis=0)
    else:
        local_count = X.shape[0]
        local_sum = np.sum(X, axis=0)
    full_count = mpiops.comm.reduce(local_count, op=mpiops.MPI.SUM, root=0)
    full_sum = mpiops.comm.reduce(local_sum, op=sum0_op, root=0)
    if mpiops.chunk_index == 0:
        centroid = full_sum / float(full_count)
    centroid = mpiops.comm.bcast(centroid, root=0)
    return centroid


def reseed_point(X, C, index):
    """ Re-initialise the centre of a class if it loses all its members

    This should almost never happen. If it does, find the point furthest
    from all the other cluster centres and use that. Maybe a bad idea but
    a decent first pass

    Parameters
    ----------
    X : ndarray
        (n, d) array of points
    C : ndarray
        (k, d) array of cluster centres
    index : int >= 0
        index between 0..k-1 of the cluster that has lost it's points

    Returns
    -------
    new_point : ndarray
        d-dimensional point for replacing the empty cluster centre.
    """
    log.info("Reseeding class with no members")
    nsplits = max(1, int(X.shape[0]/distance_partition_size))
    splits = np.array_split(X, nsplits)
    empty_index = np.ones(C.shape[0], dtype=bool)
    empty_index[index] = False
    local_candidate = None
    local_cost = 1e23
    for x_i in splits:
        D2_x = scipy.spatial.distance.cdist(x_i, C, metric='sqeuclidean')
        costs = np.sum(D2_x[:, empty_index], axis=1)
        potential_idx = np.argmax(costs)
        potential_cost = costs[potential_idx]
        if potential_cost < local_cost:
            local_candidate = x_i[potential_idx]
            local_cost = potential_cost
    best_pernode = mpiops.comm.allgather(local_cost)
    best_node = np.argmax(best_pernode)
    new_point = mpiops.comm.bcast(local_candidate, root=best_node)
    return new_point


def kmeans_step(X, C, classes, weights=None):
    """ A single step of the k-means algorithm.

    Assigns every point in X a centre, then computes the centroid of all
    x assigned to each centre, then updates that centre to be the new
    centroid.

    Parameters
    ----------
    X : ndarray
        (n, d) array of points to be clustered
    C : ndarray
        (k, d) array of initial cluster centres
    classes : ndarray
        (n,) array of initial class assignments
    weights : ndarray (optional)
        weights for points x in X that allow for different 'masses' or
        repetitions in the centroid calculation

    Returns
    -------
    C_new : ndarray
        (k, d) array of new cluster centres
    """
    C_new = np.zeros_like(C)
    for i in range(C.shape[0]):
        indices = classes == i
        n_members = mpiops.comm.allreduce(np.sum(indices), op=mpiops.MPI.SUM)
        if n_members == 0:
            C_new[i] = reseed_point(X, C, i)
        else:
            X_ind = X[indices]
            w_ind = (None if weights is None
                     else weights[indices][:, np.newaxis])
            C_new[i] = centroid(X_ind, w_ind)

    return C_new


def run_kmeans(X, C, k, weights=None, training_data=None, max_iterations=1000):
    """Cluster points into k clusters using K-means

    This is a distributed implementation of Johnson's algorithm that performs
    a convex optimization to find the locally optimal assignment of points
    and cluster centres. It depends heavily on the inital cluster centres C

    Parameters
    ----------
    X : ndarray
        (n, d) array n d-dimensional of points to cluster
    C : ndarray
        (k, d) array of initial cluster centres
    k : int > 0
        number of clusters
    weights : ndarray (optional)
        (n,) array of optional repetition weights for points in X,
        A weight of 2. implies there are 2 points at that location
    training_data : TrainingData (optional)
        An instance of the TrainingData class  containing fixed cluster
        assignments for some of the x in X
    max_iterations: int > 0 (optional)
        The algorithm will return after this many iterations, even if it
        hasn't converged

    Returns
    -------
    C : ndarray
        (k, d) array of final cluster centres, ordered (0..k-1)
    classes : ndarray
        (n,) array of class assignments (0..k-1) for each x in X
    """
    classes, cost = compute_class(X, C, training_data)
    for i in range(max_iterations):
        C_new = kmeans_step(X, C, classes, weights=weights)
        classes_new, cost = compute_class(X, C_new)
        delta_local = np.sum(classes != classes_new)
        delta = mpiops.comm.allreduce(delta_local, op=mpiops.MPI.SUM)
        log.info("kmeans it: {}\tcost:{:.3f}\tdelta: {}".format(
            i, cost, delta))
        C = C_new
        classes = classes_new
        if delta == 0:
            break
    return C, classes


def initialise_centres(X, k, l, training_data=None, max_iterations=1000):
    """
    Use Kmeans|| to find initial cluster centres

    This algorithm finds generates log(n) candidate samples efficiently,
    then uses k-means to cluster them into k initial starting centres
    used in the main algorithm (clustering X)

    Parameters
    ----------
    X : ndarray
        (n,d) array of points to cluster
    k : int > 0
        number of clusters
    l : float > 0
        Oversample factor. See weighted_starting_candidates.
    training_data : TrainingData (optional)
        Optional hard assignments of certain points in X
    max_iterations : int > 0
        The algorithm will terminate after this many iterations even
        if it hasn't converged.

    Returns
    -------
    C_init : ndarray
        (k, d) array of starting cluster centres for clustering X with k-means.
    """
    log.info("Initialising K-means centres from samples and training data")
    w, C = weighted_starting_candidates(X, k, l)
    Ck_init_indices = (np.random.choice(C.shape[0], size=k, replace=False)
                       if mpiops.chunk_index == 0 else None)
    Ck_init_indices = mpiops.comm.bcast(Ck_init_indices, root=0)
    Ck_init = C[Ck_init_indices]
    log.info("Running K-means on candidate samples")
    C_init, _ = run_kmeans(C, Ck_init, k, weights=w,
                           training_data=None,
                           max_iterations=max_iterations)

    # Force centres to use training data if available
    if training_data:
        for i in range(k):
            k_indices = training_data.classes == i
            has_training = mpiops.comm.allreduce(np.sum(k_indices),
                                                 op=mpiops.MPI.SUM) > 0
            if has_training:
                x_indices = training_data.indices[k_indices]
                X_data = X[x_indices]
                C_init[i] = centroid(X_data)
    return C_init


def compute_n_classes(classes, config):
    """The number of cluster centres to use for K-means

    Just handles the case where someone specifies k=5 but labels 10 classes
    in the training data. This will return k=10.

    Parameters
    ----------
    classes : ndarray
        an array of hard class assignments given as training data
    config : Config
        The app config class holding the number of classes asked for

    Returns
    -------
    k : int > 0
        The max of k and the number of classes referenced in the training data
    """
    k = mpiops.comm.allreduce(np.amax(classes), op=mpiops.MPI.MAX)
    k = int(max(k, config.n_classes))
    return k


def extract_data(image_source, coords_list):
    with rasterio.open(image_source) as src:
        vals = [sample[0] for sample in src.sample(coords_list)]
        vals = np.array(vals)

    return vals


def extract_features_lon_lat(main_config):
    results = []

    sample_set = False
    sample_coords = None
    for s in main_config.feature_sets:
        extracted_chunks = {}
        for tif in s.files:
            name = path.abspath(tif)
            if not sample_set:
                sample_coords = gen_sample_coords(name, main_config.subsample_fraction)

            x = extract_data(name, sample_coords)
            val_count = x.size
            x = np.reshape(x, (val_count, 1, 1, 1))
            x = ma.array(x, mask=np.zeros([val_count, 1, 1, 1]))
            # TODO this may hurt performance. Consider removal
            if type(x) is np.ma.MaskedArray:
                count = mpiops.count(x)
                missing_percent = missing_percentage(x)
                t_missing = mpiops.comm.allreduce(
                    missing_percent) / mpiops.chunks
                log.info("{}: {}px {:2.2f}% missing".format(
                    name, count, t_missing))
            extracted_chunks[name] = x
        extracted_chunks = OrderedDict(sorted(
            extracted_chunks.items(), key=lambda t: t[0]))

        results.append(extracted_chunks)

    return results


def gen_sample_coords(image_source, subsample_frac):
    print('Generating sample coords')
    with rasterio.open(image_source, r) as src:
        no_data = src.nodata
        current_data = src.read(1)
        row, col = np.where(current_data != no_data)
        num_samples = round(src.height * src.width * subsample_frac)
        random_idx = np.random.choice(row.shape[0], num_samples, replace=False)
        sample_row = list(row(random_idx))
        sample_col = list(col(random_idx))
        x_y_coords = list(zip(sample_col, sample_row))

    print()
    return x_y_coords


def center_dist_plot(dist_mat, config, current_time):
    fig, ax = plt.subplots()
    ax.matshow(dist_mat, cmap='seismic')

    for (i, j), z in np.ndenumerate(dist_mat):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    plot_name = f'cluster_center_distances_{current_time}.png'
    full_save_path = path.join(config.output_dir, plot_name)
    fig.savefig(full_save_path)


def calc_cluster_dist(centres):
    # This is not an efficient implementation
    # Will change this to be better later
    output_mat = np.zeros(centres.shape[0], centres.shape[0])
    for iy, ix in np.ndindex(output_mat.shape):
        cent_one = centres[iy, :]
        cent_two = centres[ix, :]
        current_dist = np.linalg.norm(cent_one - cent_two)
        output_mat[iy, ix] = current_dist

    return output_mat


def spatial_clusters(lon_lat, classified, config, current_time):
    fig, ax = plt.subplots()
    ax.scatter(lon_lat[1], lon_lat[0], c=classified)
    plot_name = f'cluster_spatial_plot_{current_time}.png'
    full_save_path = path.join(config.output_dir, plot_name)
    fig.savefig(full_save_path)


def feature_scatters(x_data, classified, config, current_time):
    num_plots = x_data.shape[1]
    n_row, n_col = select_subplot_grid_dims(num_plots)
    plot_width = 5 * n_col
    plot_height = 5 * n_row
    fig, axs = plt.subplots(n_row, n_col, figsize=(plot_width, plot_height), dpi=250)
    for feat_idx in range(num_plots):
        current_plot_data = x_data[:, feat_idx]
        current_ax = np.ravel(axs)[feat_idx]
        current_ax.scatter(classified, current_plot_data, c=classified)

    plot_name = f'cluster_feature_scatters_{current_time}.png'
    full_save_path = path.join(config.output_dir, plot_name)
    fig.savefig(full_save_path)


def all_feature_scatters(config, current_time):
    feat_list = []
    for s in config.feature_sets:
        for tif in s.files:
            name = path.abspath(tif)
            feat_list.append(name)

    feat_pairs = list(combinations(feat_list, 2))
    plot_idx = 0
    for pair in feat_pairs:
        feature_scatter_render(pair, config, current_time, plot_idx)
        plot_idx += 1

    print('Feature scatters complete')


def get_n_colours(n):
    return [cool(float(i) / n) for i in range(n)]


def get_min_max(data_src):
    max = None
    min = None
    for row in range(data_src.height):
        print("{:.2%}".format(row/data_src.height))
        data = data_src.read(1, window=Window(0, row, data_src.width, 1))
        if np.isnan(data_src.nodata):
            data_present = np.where(~np.isnan(data))
        else:
            data_present = np.where(data != data_src.nodata)

        data = data[data_present]
        if data.size > 0:
            current_max = np.max(data)
            if (max is None) or (current_max > max):
                max = current_max

            current_min = np.min(data)
            if (min is None) or (current_min < min):
                min = current_min

    return min, max


def feature_scatter_render(feat_pair, config, current_time, plot_idx):
    pred_file_path = path.join(config.output_dir, 'kmeans_class.tif')
    pred_src = rasterio.open(pred_file_path)
    pred_no_data = pred_src.nodata

    feat_0_src = rasterio.open(feat_pair[0])
    min_0, max_0 = get_min_max(feat_0_src)
    feat_1_src = rasterio.open(feat_pair[1])
    min_1, max_1 = get_min_max(feat_1_src)
    bound_list = [min_0-1.0, max_0+1.0, min_1-1.0, max_1+1.0]
    print(bound_list)

    fig, ax = plt.subplots(facecolor='none', dpi=500, figsize=(24, 16))
    ax.axis(bound_list)
    fig.canvas.draw()
    col = ax.scatter([5], [5], color=[0.1, 0.1, 0.1], alpha=0.3)
    num_class = config.n_classes
    colour_list = get_n_colours(num_class)
    win_height = 1
    win_width = pred_src.width
    win_col_offset = 0
    for row in range(pred_src.height):
        print("Plotting: {:.2%}".format(row/pred_src.height))
        get_win = Window(win_col_offset, row, win_width, win_height)
        pred_data = pred_src.read(1, window=get_win)
        pred_data = pred_data.reshape(pred_data.size, 1)
        data_idx = np.where(pred_data != pred_no_data)
        pred_data = pred_data[data_idx]

        feat_0_data = feat_0_src.read(1, window=get_win)
        feat_0_data = feat_0_data.reshape(feat_0_data.size, 1)
        feat_0_data = feat_0_data[data_idx]

        feat_1_data = feat_1_src.read(1, window=get_win)
        feat_1_data = feat_1_data.reshape(feat_1_data.size, 1)
        feat_1_data = feat_1_data[data_idx]

        for clust in range(num_class):
            class_idx = np.where(pred_data == float(clust))
            current_0 = feat_0_data[class_idx]
            current_1 = feat_1_data[class_idx]
            for i in range(current_0.shape[0]):
                col.set_offsets((current_0[i], current_1[i]))
                col.set_color(colour_list[clust])
                ax.draw_artist(col)

    print('Done, now saving')
    plot_name = f'feature_scatter_{plot_idx}_{current_time}.png'
    full_save_path = path.join(config.output_dir, plot_name)
    save_scatter(fig, full_save_path)
    fig.clf()


def save_scatter(fig, filename):
    """We have to work around `fig.canvas.print_png`, etc calling `draw`."""
    renderer = fig.canvas.renderer
    with open(filename, 'w') as outfile:
        _png.write_png(renderer._renderer.buffer_rgba(),
                       renderer.width, renderer.height,
                       outfile, fig.dpi)


def feature_scatter(feat_pair, config, current_time, plot_idx):
    pred_file_path = path.join(config.output_dir, 'kmeans_class.tif')
    pred_src = rasterio.open(pred_file_path)
    pred_no_data = pred_src.nodata
    feat_0_src = rasterio.open(feat_pair[0])
    feat_1_src = rasterio.open(feat_pair[1])

    # Let's assume shape match, but will add a check in
    fig, ax = plt.subplots()
    num_class = config.n_classes
    colour_list = get_n_colours(num_class)
    win_height = 1
    win_width = pred_src.width
    win_col_offset = 0
    for row in range(pred_src.height):
        print(f'Calculating row {row+1} of {pred_src.height}')
        get_win = Window(win_col_offset, row, win_width, win_height)
        pred_data = pred_src.read(1, window=get_win)
        pred_data = pred_data.reshape(pred_data.size, 1)
        data_idx = np.where(pred_data != pred_no_data)
        pred_data = pred_data[data_idx]

        feat_0_data = feat_0_src.read(1, window=get_win)
        feat_0_data = feat_0_data.reshape(feat_0_data.size, 1)
        feat_0_data = feat_0_data[data_idx]

        feat_1_data = feat_1_src.read(1, window=get_win)
        feat_1_data = feat_1_data.reshape(feat_1_data.size, 1)
        feat_1_data = feat_1_data[data_idx]

        for clust in range(num_class):
            class_idx = np.where(pred_data == float(clust))
            ax.scatter(feat_0_data[class_idx], feat_1_data[class_idx], color=colour_list[clust])

    plot_name = f'feature_scatter_{plot_idx}_{current_time}.png'
    full_save_path = path.join(config.output_dir, plot_name)
    fig.savefig(full_save_path)
    fig.clf()


def generate_save_plots(config):
    current_time = time.strftime("%Y%m%d-%H%M%S")

    # distance_mat = calc_cluster_dist(model.centres)
    # center_dist_plot(distance_mat, config, current_time)

    all_feature_scatters(config, current_time)

    print('Plotting done')
