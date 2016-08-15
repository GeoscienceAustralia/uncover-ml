import logging

import numpy as np
from mpi4py import MPI
import sklearn.datasets
import scipy.spatial
import matplotlib.pyplot as pl

log = logging.getLogger(__name__)

comm = MPI.COMM_WORLD

chunks = comm.Get_size()

chunk_index = comm.Get_rank()

n_samples = 1000 * chunks
n_features = 10
k_true = 5
k = 5
l = 100.0 / np.sqrt(chunks)
maxit = 10000
n_supervised_classes = 1
n_supervised_samples = 5

np.random.seed(1)

# TODO handle 'missing' data


def sum_axis_0(x, y, dtype):
    s = np.sum(np.vstack((x, y)), axis=0)
    return s

sum0_op = MPI.Op.Create(sum_axis_0, commute=True)


class TrainingData:
    def __init__(self, indices, classes):
        self.indices = indices
        self.classes = classes


def generate_data():
    X, y = sklearn.datasets.make_blobs(n_samples=n_samples,
                                       n_features=n_features,
                                       centers=k_true)
    # get some semi-supervised labels
    indices = []
    classes = []
    my_X = np.array_split(X, chunks)[chunk_index]
    my_y = np.array_split(y, chunks)[chunk_index]
    for k in range(n_supervised_classes):
        indices.append(np.argwhere(my_y == k)[0:n_supervised_samples])
        classes.append(np.zeros(n_supervised_samples, dtype=int) + k)
    indices = np.concatenate(indices, axis=0)
    classes = np.concatenate(classes, axis=0)
    if chunk_index == 1:  # no training data on some nodes
        indices = np.array([], dtype=int)
        classes = np.array([], dtype=int)
    training_data = TrainingData(indices, classes)
    return my_X, my_y, training_data


def kmean_distance2(x, C):
    """
    squared euclidian distance to the nearest cluster centre
    c - cluster centres
    x - nxd array of n d-dimensional points
    outputs:
    d - n length array of distances
    """
    D2_x = scipy.spatial.distance.cdist(x, C, metric='sqeuclidean')
    d2_x = np.amin(D2_x, axis=1)
    return d2_x


def compute_weights(x, C):
    """ for each c in C, return number of points in x closer to c
    than any other point in C """
    D2_x = scipy.spatial.distance.cdist(x, C, metric='sqeuclidean')
    closests = np.argmin(D2_x, axis=1)
    weights = np.bincount(closests, minlength=C.shape[0])
    return weights


def weighted_starting_candidates(X, k, l):
    # sample uniformly 1 point from X
    C = None
    if chunk_index == 0:
        idx = np.random.choice(X.shape[0])
        C = [X[idx]]
    C = comm.bcast(C, root=0)
    d2_x = kmean_distance2(X, C)
    phi_x_c_local = np.sum(d2_x)
    phi_x_c = comm.allreduce(phi_x_c_local, op=MPI.SUM)
    psi = int(round(np.log(phi_x_c)))
    for i in range(psi):
        d2_x = kmean_distance2(X, C)
        phi_x_c_local = np.sum(d2_x)
        probs = l*d2_x/phi_x_c_local
        draws = np.random.rand(probs.shape[0])
        hits = draws <= probs
        new_c = X[hits]
        C = np.concatenate([C] + comm.allgather(new_c), axis=0)

    w = compute_weights(X, C)
    return w, C


def compute_class(X, C, training_data=None):
    D2_x = scipy.spatial.distance.cdist(X, C, metric='sqeuclidean')
    classes = np.argmin(D2_x, axis=1)
    # force assignment of the training data
    if training_data:
        classes[training_data.indices] = training_data.classes
    return classes


def centroid(X, weights=None):
    centroid = np.zeros(X.shape[1])
    if weights is not None:
        local_count = np.sum(weights)
        local_sum = np.sum(X * weights, axis=0)
    else:
        local_count = X.shape[0]
        local_sum = np.sum(X, axis=0)

    full_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    full_sum = comm.reduce(local_sum, op=sum0_op, root=0)
    if chunk_index == 0:
        centroid = full_sum / full_count
    centroid = comm.bcast(centroid, root=0)
    return centroid


def kmeans_step(X, C, classes, weights=None):
    C_new = np.zeros_like(C)
    for i in range(C.shape[0]):
        indices = classes == i
        if np.sum(indices) == 0:
            raise RuntimeError("K-means error: "
                               "some centres have no closest point")
        X_ind = X[indices]
        w_ind = None if weights is None else weights[indices][:, np.newaxis]
        C_new[i] = centroid(X_ind, w_ind)

    return C_new


def run_kmeans(X, C, k, weights=None, training_data=None, max_iterations=1000):
    classes = compute_class(X, C, training_data)
    for i in range(max_iterations):
        C_new = kmeans_step(X, C, classes, weights=weights)
        classes_new = compute_class(X, C_new)
        delta_local = np.sum(classes != classes_new)
        delta = comm.allreduce(delta_local, op=MPI.SUM)
        if chunk_index == 0:
            print("kmeans it: {} delta: {}".format(i, delta))
        C = C_new
        classes = classes_new
        if delta == 0:
            break
    return C, classes


def initialise_centres(X, k, l, training_data=None):

    w, C = weighted_starting_candidates(X, k, l)
    Ck_init_indices = (np.random.choice(C.shape[0], size=k, replace=False)
                       if chunk_index == 0 else None)
    Ck_init_indices = comm.bcast(Ck_init_indices, root=0)
    Ck_init = C[Ck_init_indices]
    C_init, _ = run_kmeans(C, Ck_init, k, weights=w,
                           training_data=training_data,
                           max_iterations=maxit)

    # Force centres to use training data if available
    if training_data:
        for i in range(k):
            k_indices = training_data.classes == i
            has_training = comm.allreduce(np.sum(k_indices), op=MPI.SUM) > 0
            if has_training:
                x_indices = training_data.indices[k_indices]
                X_data = X[x_indices]
                C_init[i] = centroid(X_data)

    return C_init


def plot(X, classes, C):
    pl.figure()
    pl.scatter(X[:, 0], X[:, 1], c=classes)
    pl.plot(C[:, 0], C[:, 1], 'ro', ms=10)
    pl.show()


def main():
    if chunk_index == 0:
        print("finding initialiser set...")
    X, y, training_data = generate_data()
    # get the initial centres
    C_init = initialise_centres(X, k, l, training_data)
    # now cluster the candidates
    comm.barrier()
    if chunk_index == 0:
        print("running full k-means:")
    C_final, assignments = run_kmeans(X, C_init, k,
                                      training_data=training_data,
                                      max_iterations=maxit)
    if chunk_index == 0:
        plot(X, assignments, C_final)
if __name__ == "__main__":
    main()
