""" Scripts for validation """

from __future__ import division

import numpy as np


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
