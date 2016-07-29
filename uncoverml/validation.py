""" Scripts for validation """

from __future__ import division

import matplotlib as pl

import numpy as np

from revrand.metrics import lins_ccc, mll, msll, smse

from sklearn.metrics import explained_variance_score, r2_score

from uncoverml.models import apply_multiple_masked


metrics = {'r2_score': r2_score,
           'expvar': explained_variance_score,
           'smse': smse,
           'lins_ccc': lins_ccc,
           'mll': mll,
           'msll': msll}


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


def get_first_dim(y):
    return y[:, 0] if y.ndim > 1 else y


# Decorator to deal with probabilistic output for non-probabilistic scores
def score_first_dim(func):
    def newscore(y_true, y_pred, *args, **kwargs):
        return func(y_true.flatten(), get_first_dim(y_pred), *args, **kwargs)
    return newscore


def calculate_validation_scores(ys, yt, eys):

    probscores = ['msll', 'mll']

    scores = {}
    for m in metrics:

        if m not in probscores:
            score = apply_multiple_masked(score_first_dim(metrics[m]),
                                          (ys, eys))
        elif eys.ndim == 2:
            if m == 'mll' and eys.shape[1] > 1:
                score = apply_multiple_masked(mll, (ys, eys[:, 0], eys[:, 1]))
            elif m == 'msll' and eys.shape[1] > 1:
                score = apply_multiple_masked(msll, (ys, eys[:, 0], eys[:, 1]),
                                              (yt,))
            else:
                continue
        else:
            continue

        scores[m] = score
    return scores


def y_y_plot(y1, y2, y_label=None, y_exp_label=None, title=None,
             outfile=None, display=None):
    fig = pl.figure()
    maxy = max(y1.max(), get_first_dim(y2).max())
    miny = min(y1.min(), get_first_dim(y2).min())
    apply_multiple_masked(pl.plot, (y1, get_first_dim(y2)), ('k.',))
    pl.plot([miny, maxy], [miny, maxy], 'r')
    pl.grid(True)
    pl.xlabel(y_label)
    pl.ylabel(y_exp_label)
    pl.title(title)
    if outfile is not None:
        fig.savefig(outfile + ".png")
    if display:
        pl.show()
