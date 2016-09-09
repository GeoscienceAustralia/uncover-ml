import logging

import numpy as np
import scipy.spatial

from uncoverml import mpiops
from uncoverml import geoio
from uncoverml import features

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

    def predict(self, x):
        y_star, _ = compute_class(x, self.centres)
        # y_star = y_star[:, np.newaxis].astype(float)
        y_star = y_star.astype(float)
        return y_star

    def get_predict_tags(self):
        tags = ['class']
        return tags


def kmean_distance2(x, C):
    """
    squared euclidian distance to the nearest cluster centre
    c - cluster centres
    x - nxd array of n d-dimensional points
    outputs:
    d - n length array of distances
    """
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
    """ for each c in C, return number of points in x closer to c
    than any other point in C """
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
    # sample uniformly 1 point from X
    C = None
    if mpiops.chunk_index == 0:
        idx = np.random.choice(X.shape[0])
        C = [X[idx]]
    C = mpiops.comm.bcast(C, root=0)
    d2_x = kmean_distance2(X, C)
    phi_x_c_local = np.sum(d2_x)
    phi_x_c = mpiops.comm.allreduce(phi_x_c_local, op=mpiops.MPI.SUM)
    psi = int(round(np.log(phi_x_c)))
    log.info("kmeans|| using {} sampling iterations".format(psi))
    for i in range(psi):
        d2_x = kmean_distance2(X, C)
        phi_x_c_local = np.sum(d2_x)
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
    """ find the point furthest away from the the current centres"""
    log.info("Reseeding class with no members")
    idx = np.ones(C.shape[0], dtype=bool)
    idx[index] = False
    D2_x = scipy.spatial.distance.cdist(X, C, metric='sqeuclidean')
    costs = np.sum(D2_x[:, idx], axis=1)
    local_candidate = np.argmax(costs)
    local_cost = costs[local_candidate]
    best_pernode = mpiops.comm.allgather(local_cost)
    best_node = np.argmax(best_pernode)
    new_point = mpiops.comm.bcast(X[local_candidate], root=best_node)
    return new_point


def kmeans_step(X, C, classes, weights=None):
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
    classes, cost = compute_class(X, C, training_data)
    for i in range(max_iterations):
        C_new = kmeans_step(X, C, classes, weights=weights)
        classes_new, cost = compute_class(X, C_new)
        delta_local = np.sum(classes != classes_new)
        delta = mpiops.comm.allreduce(delta_local, op=mpiops.MPI.SUM)
        if mpiops.chunk_index == 0:
            log.info("kmeans it: {}\tcost:{:.3f}\tdelta: {}".format(
                i, cost, delta))
        C = C_new
        classes = classes_new
        if delta == 0:
            break
    return C, classes


def initialise_centres(X, k, l, training_data=None, max_iterations=1000):
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
    k = mpiops.comm.allreduce(np.amax(classes), op=mpiops.MPI.MAX)
    k = max(k, config.n_classes)
    return k


