""" Scripts for validation """

from __future__ import division

import numpy as np


from revrand.metrics import smse, mll, msll, lins_ccc
from sklearn.metrics import r2_score, explained_variance_score


def get_first_dim(Y):

    return Y[:, 0] if Y.ndim > 1 else Y


# Decorator to deal with probabilistic output for non-probabilistic scores
def score_first_dim(func):

    def newscore(y_true, y_pred, *args, **kwargs):

        return func(y_true.flatten(), get_first_dim(y_pred), *args, **kwargs)

    return newscore

metrics = {'r2_score': r2_score,
           'expvar': explained_variance_score,
           'smse': smse,
           'lins_ccc': lins_ccc,
           'mll': mll,
           'msll': msll
           }

probscores = ['msll', 'mll']


#
# Data Partitioning
#

def gen_cfold_data(X, Y, k=5):
    """
    Generator to divide a dataset k non-overlapping folds.

    Parameters
    ----------
        X: ndarray
            (D, N) array where D is the dimensionality, and N is the number
            of samples (X can also be a 1-d vector).
        Y: ndarray
            (N,) training target data vector of length N.
        k: int, optional
            the number of folds for testing and training.

    Yields
    ------
        Xr: ndarray
            (D, ((k-1) * N / k)) array of training input data
        Yr: ndarray
            ((k-1) * N / k,) array of training target data
        Xs: ndarray
            (D, N / k) array of testing input data
        Ys: ndarray
            (N / k,) array of testing target data

    Note
    ----
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


# def gen_cfold_ind(nsamples, k=5, seed=None):
#     """
#     Generator to return random test and training indices for cross fold
#     validation.

#     Parameters
#     ----------
#         nsamples: int
#             the number of samples in the dataset
#         k: int, optional
#             the number of folds
#         seed: int, optional
#             random seed for numpy permutation

#     Yields
#     ------
#         rind: ndarray
#             training indices of shape (nsamples * (k-1)/k,)
#         sind: ndarray
#             testing indices of shape (nsamples * 1/k,)

#     Note
#     ----
#         Each call to this generator returns a random but non-overlapping
#         split of data.

#     """

#     cvinds, _ = split_cfold(nsamples, k, seed)

#     for i in range(k):
#         sind = cvinds[i]
#         rind = np.concatenate(cvinds[0:i] + cvinds[i + 1:])
#         yield (rind, sind)


def split_cfold(nsamples, k=5, seed=None):
    """
    Function that returns indices for splitting data into random folds.

    Parameters
    ----------
        nsamples: int
            the number of samples in the dataset
        k: int, optional
            the number of folds
        seed: int, optional
            random seed to provide to numpy

    Returns
    -------
        cvinds: list
            list of arrays of length k, each with approximate shape (nsamples /
            k,) of indices. These indices are randomly permuted (without
            replacement) of assignments to each fold.
        cvassigns: ndarray
            array of shape (nsamples,) with each element in [0, k), that can be
            used to assign data to a fold. This corresponds to the indices of
            cvinds.

    """
    np.random.seed(seed)
    pindeces = np.random.permutation(nsamples)
    cvinds = np.array_split(pindeces, k)

    cvassigns = np.zeros(nsamples, dtype=int)
    for n, inds in enumerate(cvinds):
        cvassigns[inds] = n

    return cvinds, cvassigns
