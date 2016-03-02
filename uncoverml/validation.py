""" Scripts for validation """

import numpy as np


def k_fold_CV(X, Y, k=5):
    """ Generator to divide a dataset k non-overlapping folds.

        Parameters
        ----------
            X: ndarray
                (D, N) array where D is the dimensionality, and N is the number
                of samples (X can also be a 1-d vector).
            Y: ndarray
                (N,) training target data vector of length N.
            k: int, optional
                the number of folds for testing and training.

        Yeilds
        ------
            Xr: ndarray
                (D, ((k-1) * N / k)) array of training input data
            Yr: ndarray
                ((k-1) * N / k,) array of training target data
            Xs: [D x (N / k)] testing input data
            Ys: [N / k] testing output data

            All of these are randomly split (but non-overlapping per call)
    """

    X = np.atleast_2d(X)
    random_indices = np.random.permutation(X.shape[1])
    X = X[:, random_indices]
    Y = Y[random_indices]
    X_groups = np.array_split(X, k, axis=1)
    Y_groups = np.array_split(Y, k)

    for i in range(k):
        X_s = X_groups[i]
        Y_s = Y_groups[i]
        X_r = np.hstack(X_groups[0:i] + X_groups[i + 1:])
        Y_r = np.concatenate(Y_groups[0:i] + Y_groups[i + 1:])
        yield (X_r, Y_r, X_s, Y_s)


def k_fold_CV_ind(nsamples, k=5):
    """ Generator to return random test and training indices for cross fold
        validation.

        Arguments:
            nsamples: the number of samples in the dataset
            k: [optional] the number of folds

        Returns:
            rind: training indices of length nsamples * (k-1)/k
            sind: testing indices of length nsamples * 1/k

            Each call to this generator returns a random but non-overlapping
            split of data.
    """

    pindeces = np.random.permutation(nsamples)
    pgroups = np.array_split(pindeces, k)

    for i in range(k):
        sind = pgroups[i]
        rind = np.concatenate(pgroups[0:i] + pgroups[i + 1:])
        yield (rind, sind)
