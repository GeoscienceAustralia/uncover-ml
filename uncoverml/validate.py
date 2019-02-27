""" Scripts for validation """

from __future__ import division
import logging
import copy

import numpy as np
from sklearn.metrics import (explained_variance_score, r2_score,
                             accuracy_score, log_loss, roc_auc_score,
                             confusion_matrix)
from revrand.metrics import lins_ccc, mll, smse

from uncoverml.models import apply_multiple_masked
from uncoverml import mpiops
from uncoverml import predict
from uncoverml import features as feat
from uncoverml import targets as targ
from uncoverml.learn import all_modelmaps as modelmaps


log = logging.getLogger(__name__)


MINPROB = 1e-5  # Numerical guard for log-loss evaluation

regression_metrics = {
    'r2_score': lambda y, py, vy, y_t, py_t, vy_t:  r2_score(y, py),
    'expvar': lambda y, py, vy, y_t, py_t, vy_t:
    explained_variance_score(y, py),
    'smse': lambda y, py, vy, y_t, py_t, vy_t: smse(y, py),
    'lins_ccc': lambda y, py, vy, y_t, py_t, vy_t: lins_ccc(y, py),
    'mll': lambda y, py, vy, y_t, py_t, vy_t: mll(y, py, vy)
}


transformed_regression_metrics = {
    'r2_score_transformed': lambda y, py, vy, y_t, py_t, vy_t:
    r2_score(y_t, py_t),
    'expvar_transformed': lambda y, py, vy, y_t, py_t, vy_t:
    explained_variance_score(y_t, py_t),
    'smse_transformed': lambda y, py, vy, y_t, py_t, vy_t: smse(y_t, py_t),
    'lins_ccc_transformed': lambda y, py, vy, y_t, py_t, vy_t: lins_ccc(y_t,
                                                                        py_t),
    'mll_transformed': lambda y, py, vy, y_t, py_t, vy_t: mll(y_t, py_t, vy_t)
}


def _binarizer(y, p, func, **kwargs):
    yb = np.zeros_like(p)
    n = len(y)
    yb[range(n), y.astype(int)] = 1.
    score = func(yb, p, **kwargs)
    return score


classification_metrics = {
    'accuracy': lambda y, ey, p: accuracy_score(y, ey),
    'log_loss': lambda y, ey, p: log_loss(y, p),
    'auc': lambda y, ey, p: _binarizer(y, p, roc_auc_score, average='macro'),
    'confusion': lambda y, ey, p: (confusion_matrix(y, ey) / len(y)).tolist()
}


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
    rnd = np.random.RandomState(seed)
    pindeces = rnd.permutation(nsamples)
    cvinds = np.array_split(pindeces, k)

    cvassigns = np.zeros(nsamples, dtype=int)
    for n, inds in enumerate(cvinds):
        cvassigns[inds] = n

    return cvinds, cvassigns


def classification_validation_scores(ys, eys, pys):
    """ Calculates the validation scores for a regression prediction
    Given the test and training data, as well as the outputs from every model,
    this function calculates all of the applicable metrics in the following
    list, and returns a dictionary with the following (possible) keys:
        + accuracy
        + log_loss
        + f1

    Parameters
    ----------
    ys: numpy.array
        The test data outputs, one-hot representation
    eys: numpy.array
        The (hard) predictions made by the trained model on test data, one-hot
        representation
    pys: numpy.array
        The probabilistic predictions made by the trained model on test data

    Returns
    -------
    scores: dict
        A dictionary containing all of the evaluated scores.
    """
    scores = {}
    # in case we get hard probabilites and log freaks out
    pys = np.minimum(np.maximum(pys, MINPROB), 1. - MINPROB)

    for k, m in classification_metrics.items():
        scores[k] = apply_multiple_masked(m, (ys, eys, pys))

    return scores


def regression_validation_scores(y, ey, model):
    """ Calculates the validation scores for a regression prediction
    Given the test and training data, as well as the outputs from every model,
    this function calculates all of the applicable metrics in the following
    list, and returns a dictionary with the following (possible) keys:
        + r2_score
        + expvar
        + smse
        + lins_ccc
        + mll
        + msll

    Parameters
    ----------
    y: numpy.array
        The test data outputs
    ey: numpy.array
        The predictions made by the trained model on test data

    Returns
    -------
    scores: dict
        A dictionary containing all of the evaluated scores.
    """
    scores = {}

    result_tags = model.get_predict_tags()

    if 'Variance' in result_tags:
        py, vy = ey[:, 0], ey[:, 1]
    else:
        py, vy = ey[:, 0], ey[:, 0]
        # don't calculate mll when variance is not available
        regression_metrics.pop('mll', None)
        transformed_regression_metrics.pop('mll_transformed', None)

    if hasattr(model, '_notransform_predict'):  # is a transformed model

        y_t = model.target_transform.transform(y)  # transformed targets
        py_t = model.target_transform.transform(py)  # transformed prediction

        regression_metrics.update(transformed_regression_metrics)

        if 'Variance' in result_tags:
            # transformed standard dev
            v_t = model.target_transform.transform(np.sqrt(vy))
            vy_t = np.square(v_t)  # transformed variances
        else:
            vy_t = py
    else:  # don't calculate if Transformed Prediction is not available
        y_t = y
        py_t = py
        vy_t = py

    for k, m in regression_metrics.items():
        scores[k] = apply_multiple_masked(m, (y, py, vy, y_t, py_t, vy_t))

    return scores


class CrossvalInfo:
    def __init__(self, scores, y_true, y_pred, classification):
        self.scores = scores
        self.y_true = y_true
        self.y_pred = y_pred
        self.classification = classification


def local_rank_features(image_chunk_sets, transform_sets, targets, config):
    """ Ranks the importance of the features based on their performance.
    This function trains and cross-validates a model with each individual
    feature removed and then measures the performance of the model with that
    feature removed. The most important feature is the one which; when removed,
    causes the greatest degradation in the performance of the model.

    Parameters
    ----------
    image_chunk_sets: dict
        A dictionary used to get the set of images to test on.
    transform_sets: list
        A dictionary containing the applied transformations
    targets: instance of geoio.Targets class
        The targets used in the cross validation
    config: config class instance
        The global config file
    """

    feature_scores = {}

    # Get all the images
    all_names = []
    for c in image_chunk_sets:
        all_names.extend(list(c.keys()))
    all_names = sorted(list(set(all_names)))  # make unique

    if len(all_names) <= 1:
        raise ValueError("Cannot perform feature ranking with only one "
                         "feature! Try turning off the 'feature_rank' option.")

    for name in all_names:
        transform_sets_leaveout = copy.deepcopy(transform_sets)
        final_transform_leaveout = copy.deepcopy(config.final_transform)
        image_chunks_leaveout = [copy.copy(k) for k in image_chunk_sets]

        for i, c in enumerate(image_chunks_leaveout):
            if name in c:
                c.pop(name)
            # if only one covariate of a feature type, delete
            # this feature type, and transformset
            if not c:
                image_chunks_leaveout.pop(i)
                transform_sets_leaveout.pop(i)

        fname = name.rstrip(".tif")
        log.info("Computing {} feature importance of {}"
                 .format(config.algorithm, fname))
        x, keep = feat.transform_features(image_chunks_leaveout,
                                          transform_sets_leaveout,
                                          final_transform_leaveout,
                                          config)
        x_all = feat.gather_features(x[keep], node=0)
        targets_all = targ.gather_targets_main(targets, keep, node=0)
        results = local_crossval(x_all, targets_all, config)
        feature_scores[fname] = results

    # Get the different types of score from one of the outputs
    if mpiops.chunk_index == 0:
        measures = list(next(feature_scores.values().__iter__()).scores.keys())
        features = sorted(feature_scores.keys())
        scores = np.empty((len(measures), len(features)))
        for m, measure in enumerate(measures):
            for f, feature in enumerate(features):
                scores[m, f] = feature_scores[feature].scores[measure]
        return measures, features, scores
    else:
        return None, None, None


def _join_dicts(dicts):
    if dicts is None:
        return
    d = {k: v for D in dicts for k, v in D.items()}
    return d


def local_crossval(x_all, targets_all, config):
    """ Performs K-fold cross validation to test the applicability of a model.
    Given a set of inputs and outputs, this function will evaluate the
    effectiveness of a model at predicting the targets, by splitting all of
    the known data. A model is trained on a subset of the total data, and then
    this model is used to predict all of the unseen targets, its performance
    can provide a benchmark to evaluate the effectiveness of a model.

    Parameters
    ----------
    x_all: numpy.array
        A 2D array containing all of the training inputs
    targets_all: numpy.array
        A 1D vector containing all of the training outputs
    config: dict
        The global config object, which is used to choose the model to train.

    Return
    ------
    result: dict
        A dictionary containing all of the cross validation metrics, evaluated
        on the unseen data subset.
    """
    # run cross validation in parallel, but one thread for each fold
    if config.multicubist or config.multirandomforest:
        config.algorithm_args['parallel'] = False

    if (mpiops.chunk_index != 0) and (not config.parallel_validate):
        return

    log.info("Validating with {} folds".format(config.folds))
    model = modelmaps[config.algorithm](**config.algorithm_args)
    classification = hasattr(model, 'predict_proba')
    y = targets_all.observations
    lon_lat = targets_all.positions
    _, cv_indices = split_cfold(y.shape[0], config.folds, config.crossval_seed)

    # Split folds over workers
    fold_list = np.arange(config.folds)
    if config.parallel_validate:
        fold_node = \
            np.array_split(fold_list, mpiops.chunks)[mpiops.chunk_index]
    else:
        fold_node = fold_list

    y_pred = {}
    y_true = {}
    fold_scores = {}

    # Train and score on each fold
    for fold in fold_node:

        print("Training fold {} of {} using process {}".format(
            fold + 1, config.folds, mpiops.chunk_index))
        train_mask = cv_indices != fold
        test_mask = ~ train_mask

        y_k_train = y[train_mask]
        lon_lat_train = lon_lat[train_mask]
        lon_lat_test = lon_lat[test_mask]

        # Extra fields
        fields_train = {f: v[train_mask]
                        for f, v in targets_all.fields.items()}
        fields_pred = {f: v[test_mask] for f, v in targets_all.fields.items()}

        # Train on this fold
        x_train = x_all[train_mask]
        apply_multiple_masked(model.fit, data=(x_train, y_k_train),
                              kwargs={'fields': fields_train,
                                      'lon_lat': lon_lat_train})

        # Testing
        y_k_pred = predict.predict(x_all[test_mask], model,
                                   fields=fields_pred,
                                   lon_lat=lon_lat_test)

        y_pred[fold] = y_k_pred

        # Regression
        if not classification:
            y_k_test = y[test_mask]
            y_true[fold] = y_k_test
            fold_scores[fold] = regression_validation_scores(
                y_k_test, y_k_pred, model)

        # Classification
        else:
            y_k_test = model.le.transform(y[test_mask])
            y_true[fold] = y_k_test
            y_k_hard, p_k = y_k_pred[:, 0], y_k_pred[:, 1:]
            fold_scores[fold] = classification_validation_scores(
                y_k_test, y_k_hard, p_k
            )

    if config.parallel_validate:
        y_pred = _join_dicts(mpiops.comm.gather(y_pred, root=0))
        y_true = _join_dicts(mpiops.comm.gather(y_true, root=0))
        scores = _join_dicts(mpiops.comm.gather(fold_scores, root=0))
    else:
        scores = fold_scores

    result = None
    if mpiops.chunk_index == 0:
        y_true = np.concatenate([y_true[i] for i in range(config.folds)])
        y_pred = np.concatenate([y_pred[i] for i in range(config.folds)])
        valid_metrics = scores[0].keys()
        scores = {m: np.mean([d[m] for d in scores.values()], axis=0)
                  for m in valid_metrics}
        score_string = "Validation complete:\n"
        for metric, score in scores.items():
            score_string += "{}\t= {}\n".format(metric, score)
        log.info(score_string)

        result_tags = model.get_predict_tags()
        y_pred_dict = dict(zip(result_tags, y_pred.T))
        result = CrossvalInfo(scores, y_true, y_pred_dict, classification)

    # change back to parallel
    if config.multicubist or config.multirandomforest:
        config.algorithm_args['parallel'] = True

    return result
