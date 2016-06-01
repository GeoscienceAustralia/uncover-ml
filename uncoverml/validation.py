""" Scripts for validation """

from __future__ import division

import numpy as np

from uncoverml.geoio import points_from_hdf, points_to_hdf, indices_from_hdf


#
# Read and write validation files
#


def output_cvindex(index, lonlat, outfile):

    points_to_hdf(lonlat, outfile, "FoldIndices", index)


def input_cvindex(cvindex_file, return_lonlat=False):

    lonlat, cv_ind = points_from_hdf(cvindex_file, "FoldIndices")

    return (cv_ind.flatten(), lonlat) if return_lonlat else cv_ind.flatten()


def output_targets(target, lonlat, outfile):

    points_to_hdf(lonlat, outfile, "targets", target)


def input_targets(target_file):

    lonlat, targets = points_from_hdf(target_file, "targets")
    target_indices = indices_from_hdf(target_file)

    result = lonlat, targets.flatten(), target_indices
    return result


def chunk_cvindex(cvind, nchunks):

    return np.array_split(cvind, nchunks)


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


def gen_cfold_ind(nsamples, k=5):
    """
    Generator to return random test and training indices for cross fold
    validation.

    Parameters
    ----------
        nsamples: int
            the number of samples in the dataset
        k: int, optional
            the number of folds

    Yields
    ------
        rind: ndarray
            training indices of shape (nsamples * (k-1)/k,)
        sind: ndarray
            testing indices of shape (nsamples * 1/k,)

    Note
    ----
        Each call to this generator returns a random but non-overlapping
        split of data.

    """

    cvinds, _ = split_cfold(nsamples, k)

    for i in range(k):
        sind = cvinds[i]
        rind = np.concatenate(cvinds[0:i] + cvinds[i + 1:])
        yield (rind, sind)


def split_cfold(nsamples, k=5):
    """
    Function that returns indices for splitting data into random folds.

    Parameters
    ----------
        nsamples: int
            the number of samples in the dataset
        k: int, optional
            the number of folds

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

    pindeces = np.random.permutation(nsamples)
    cvinds = np.array_split(pindeces, k)

    cvassigns = np.zeros(nsamples, dtype=int)
    for n, inds in enumerate(cvinds):
        cvassigns[inds] = n

    return cvinds, cvassigns
