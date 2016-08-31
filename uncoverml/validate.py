""" Scripts for validation """

from __future__ import division

import matplotlib as pl
import numpy as np

from sklearn.metrics import explained_variance_score, r2_score
from revrand.metrics import lins_ccc, mll, msll, smse

from uncoverml.models import apply_multiple_masked


metrics = {'r2_score': r2_score,
           'expvar': explained_variance_score,
           'smse': smse,
           'lins_ccc': lins_ccc,
           'mll': mll,
           'msll': msll}

lower_is_better = ['mll', 'msll', 'smse']


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



class CrossvalInfo:
    def __init__(self, scores, y_true, y_pred):
        self.scores = scores
        self.y_true = y_true
        self.y_pred = y_pred

def _join_dicts(dicts):
    if dicts is None:
        return
    d = {k: v for D in dicts for k, v in D.items()}
    return d


def local_crossval(x_all, targets_all, config):
    log.info("Validating with {} folds".format(config.folds))
    model = modelmaps[config.algorithm](**config.algorithm_args)
    y = targets_all.observations
    _, cv_indices = split_cfold(y.shape[0], config.folds, config.crossval_seed)

    # Split folds over workers
    fold_list = np.arange(config.folds)
    fold_node = np.array_split(fold_list, mpiops.chunks)[mpiops.chunk_index]

    y_pred = {}
    y_true = {}
    fold_scores = {}

    # Train and score on each fold
    for fold in fold_node:

        train_mask = cv_indices != fold
        test_mask = ~ train_mask

        y_k_train = y[train_mask]

        # Extra fields
        fields_train = {f: v[train_mask]
                        for f, v in targets_all.fields.items()}
        fields_pred = {f: v[test_mask] for f, v in targets_all.fields.items()}

        # Train on this fold
        apply_multiple_masked(model.fit, data=(x_all[train_mask], y_k_train),
                              kwargs={'fields': fields_train})

        # Testing
        y_k_pred = predict(x_all[test_mask], model, fields=fields_pred)
        y_k_test = y[test_mask]
        y_pred[fold] = y_k_pred
        y_true[fold] = y_k_test

        fold_scores[fold] = calculate_validation_scores(y_k_test,
                                                        y_k_train,
                                                        y_k_pred)

    y_pred = _join_dicts(mpiops.comm.gather(y_pred, root=0))
    y_true = _join_dicts(mpiops.comm.gather(y_true, root=0))
    scores = _join_dicts(mpiops.comm.gather(fold_scores, root=0))

    result = None
    if mpiops.chunk_index == 0:
        y_true = np.concatenate([y_true[i] for i in range(config.folds)])
        y_pred = np.concatenate([y_pred[i] for i in range(config.folds)])
        valid_metrics = scores[0].keys()
        scores = {m: np.mean([d[m] for d in scores.values()])
                  for m in valid_metrics}
        score_string = "Validation complete:\n"
        for metric, score in scores.items():
            score_string += "{}\t= {}\n".format(metric, score)
        log.info(score_string)

        result_tags = model.get_predict_tags()
        y_pred_dict = dict(zip(result_tags, y_pred.T))
        result = CrossvalInfo(scores, y_true, y_pred_dict)
    result = mpiops.comm.bcast(result, root=0)
    return result

