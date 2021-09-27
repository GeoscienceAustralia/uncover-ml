""" Scripts for validation """

from __future__ import division
import logging
import copy
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import (explained_variance_score, r2_score,
                             accuracy_score, log_loss, roc_auc_score,
                             confusion_matrix)
import eli5
from eli5.sklearn import PermutationImportance
from revrand.metrics import lins_ccc, mll, smse

from uncoverml.geoio import CrossvalInfo
from uncoverml.models import apply_multiple_masked
from uncoverml import mpiops
from uncoverml import predict, geoio
from uncoverml import features as feat
from uncoverml import targets as targ
from uncoverml.config import Config
from uncoverml.transforms.target import Identity
from uncoverml.learn import all_modelmaps as modelmaps
from uncoverml.optimise.models import transformed_modelmaps


log = logging.getLogger(__name__)


MINPROB = 1e-5  # Numerical guard for log-loss evaluation

regression_metrics = {
    'r2_score': lambda y, py, vy, ws, y_t, py_t, vy_t:  r2_score(y, py, sample_weight=ws),
    'expvar': lambda y, py, vy, ws, y_t, py_t, vy_t:
    explained_variance_score(y, py, sample_weight=ws),
    'smse': lambda y, py, vy, ws, y_t, py_t, vy_t: smse(y, py),
    'lins_ccc': lambda y, py, vy, ws, y_t, py_t, vy_t: lins_ccc(y, py),
    'mll': lambda y, py, vy, ws, y_t, py_t, vy_t: mll(y, py, vy)
}


transformed_regression_metrics = {
    'r2_score_transformed': lambda y, py, vy, ws, y_t, py_t, vy_t:
    r2_score(y_t, py_t, sample_weight=ws),
    'expvar_transformed': lambda y, py, vy, ws, y_t, py_t, vy_t:
    explained_variance_score(y_t, py_t, sample_weight=ws),
    'smse_transformed': lambda y, py, vy, ws, y_t, py_t, vy_t: smse(y_t, py_t),
    'lins_ccc_transformed': lambda y, py, vy, ws, y_t, py_t, vy_t: lins_ccc(y_t,
                                                                        py_t),
    'mll_transformed': lambda y, py, vy, ws, y_t, py_t, vy_t: mll(y_t, py_t, vy_t)
}


def _binarizer(y, p, ws, func, **kwargs):
    yb = np.zeros_like(p)
    n = len(y)
    yb[range(n), y.astype(int)] = 1.
    score = func(yb, p, sample_weight=ws, **kwargs)
    return score


classification_metrics = {
    'accuracy': lambda y, ey, ws, p: accuracy_score(y, ey, sample_weight=ws),
    'log_loss': lambda y, ey, ws, p: log_loss(y, p, sample_weight=ws),
    'auc': lambda y, ey, ws, p: _binarizer(y, p, ws, roc_auc_score, average='macro'),
    'mean_confusion': lambda y, ey, ws, p: (confusion_matrix(y, ey, sample_weight=ws)).tolist(),
    'mean_confusion_normalized': lambda y, ey, ws, p:
        (confusion_matrix(y, ey, sample_weight=ws) / len(y)).tolist()
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


def split_gfold(groups, k=5, seed=None):
    """
    Function that returns indices for splitting data into random folds respecting groups.

    Parameters
    ----------
    targets: int
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
    fold = GroupKFold(n_splits=k) if len(np.unique(groups)) >= k else \
        KFold(n_splits=k, shuffle=True, random_state=seed)
    cvinds = []
    cvassigns = np.zeros_like(groups, dtype=int)
    fold_str = fold.__class__.__name__
    for n, g in enumerate(fold.split(np.arange(groups.shape[0]), groups=groups)):
        log.info(f"{fold_str} resulted in {g[1].shape[0]} targets in Group {n}")
        cvinds.append(g[1])
        cvassigns[g[1]] = n

    return cvinds, cvassigns


def classification_validation_scores(ys, eys, ws, pys):
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
    ws: numpy.array
        The weights of the test data
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
        scores[k] = apply_multiple_masked(m, (ys, eys, ws, pys))

    return scores


def regression_validation_scores(y, ey, ws, model):
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
    ws: numpy.array
        The weights of the test data

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

    if hasattr(model, '_notransform_predict') and not isinstance(model.target_transform, Identity):  #
        # is a transformed model
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
        scores[k] = apply_multiple_masked(m, (y, py, vy, ws, y_t, py_t, vy_t))

    return scores


def permutation_importance(model, x_all, targets_all, config: Config):
    log.info("Computing permutation importance!!")
    if config.algorithm not in transformed_modelmaps.keys():
        raise AttributeError("Only the following can be used for permutation "
                             "importance {}".format(
            list(transformed_modelmaps.keys())))

    y = targets_all.observations

    classification = hasattr(model, 'predict_proba')

    if not classification:
        for score in ['explained_variance',
                      'r2',
                      'neg_mean_absolute_error',
                      'neg_mean_squared_error']:
            pi_cv = apply_multiple_masked(
                PermutationImportance(model, scoring=score,
                                      cv='prefit', n_iter=10,
                                      refit=False).fit,
                data=(x_all, y)
            )
            feature_names = geoio.feature_names(config)
            df_picv = eli5.explain_weights_df(
                pi_cv, feature_names=feature_names, top=100)
            csv = Path(config.output_dir).joinpath(
                config.name + "_permutation_importance_{}.csv".format(
                    score)).as_posix()
            df_picv.to_csv(csv, index=False)


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


def local_crossval(x_all, targets_all: targ.Targets, config: Config):
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
    w = targets_all.weights
    lon_lat = targets_all.positions
    groups = targets_all.groups

    if (len(np.unique(groups)) + 1 < config.folds) and config.group_targets:
        raise ValueError(f"Cannot continue cross-validation with chosen params as num of groups {max(groups) + 1} "
                         f"in data is less than the number of folds {config.folds}")

    _, cv_indices = split_gfold(groups, config.folds, config.crossval_seed)

    # Split folds over workers
    fold_list = np.arange(config.folds)
    if config.parallel_validate:
        fold_node = np.array_split(fold_list, mpiops.chunks)[mpiops.chunk_index]
    else:
        fold_node = fold_list

    y_pred = {}
    y_true = {}
    weight = {}
    lon_lat_ = {}
    fold_scores = {}

    # Train and score on each fold
    for fold in fold_node:
        model = modelmaps[config.algorithm](**config.algorithm_args)

        print("Training fold {} of {} using process {}".format(
            fold + 1, config.folds, mpiops.chunk_index))
        train_mask = cv_indices != fold
        test_mask = ~ train_mask

        y_k_train = y[train_mask]
        w_k_train = w[train_mask]
        lon_lat_train = lon_lat[train_mask, :]
        lon_lat_test = lon_lat[test_mask, :]

        # Extra fields
        fields_train = {f: v[train_mask]
                        for f, v in targets_all.fields.items()}
        fields_pred = {f: v[test_mask] for f, v in targets_all.fields.items()}

        # Train on this fold
        x_train = x_all[train_mask]
        apply_multiple_masked(model.fit, data=(x_train, y_k_train),
                              ** {'fields': fields_train,
                                  'sample_weight': w_k_train,
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
            w_k_test = w[test_mask]
            weight[fold] = w_k_test
            lon_lat_[fold] = lon_lat_test
            fold_scores[fold] = regression_validation_scores(y_k_test, y_k_pred, w_k_test, model)

        # Classification
        else:
            y_k_test = model.le.transform(y[test_mask])
            y_true[fold] = y_k_test
            w_k_test = w[test_mask]
            weight[fold] = w_k_test
            lon_lat_[fold] = lon_lat_test
            y_k_hard, p_k = y_k_pred[:, 0], y_k_pred[:, 1:]
            fold_scores[fold] = classification_validation_scores(y_k_test, y_k_hard, w_k_test, p_k)

    if config.parallel_validate:
        y_pred = _join_dicts(mpiops.comm.gather(y_pred, root=0))
        lon_lat_ = _join_dicts(mpiops.comm.gather(lon_lat_, root=0))
        y_true = _join_dicts(mpiops.comm.gather(y_true, root=0))
        weight = _join_dicts(mpiops.comm.gather(weight, root=0))
        scores = _join_dicts(mpiops.comm.gather(fold_scores, root=0))
    else:
        scores = fold_scores

    result = None
    if mpiops.chunk_index == 0:
        y_true = np.concatenate([y_true[i] for i in range(config.folds)])
        weight = np.concatenate([weight[i] for i in range(config.folds)])
        lon_lat = np.concatenate([lon_lat_[i] for i in range(config.folds)])
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
        if hasattr(model, '_notransform_predict'):
            y_pred_dict['transformedpredict'] = \
                model.target_transform.transform(y_pred[:, 0])
        result = CrossvalInfo(scores, y_true, y_pred_dict, weight, lon_lat, classification)

    # change back to parallel
    if config.multicubist or config.multirandomforest:
        config.algorithm_args['parallel'] = True

    return result


def plot_feature_importance(model, x_all, targets_all, conf: Config):
    log.info("Computing permutation importance!!")
    if conf.algorithm not in transformed_modelmaps.keys():
        raise AttributeError("Only the following can be used for permutation "
                             "importance {}".format(
            list(transformed_modelmaps.keys())))

    y = targets_all.observations

    classification = hasattr(model, 'predict_proba')

    if not classification:
        pi_cv = apply_multiple_masked(
            PermutationImportance(model, scoring=score,
                                  cv='prefit', n_iter=10,
                                  refit=False).fit,
            data=(x_all, y),
            model=model
        )
        feature_names = geoio.feature_names(conf)
        df_picv = eli5.explain_weights_df(
            pi_cv, feature_names=feature_names, top=100)
        csv = Path(config.output_dir).joinpath(
            config.name + "_permutation_importance_{}.csv".format(
                score)).as_posix()
        df_picv.to_csv(csv, index=False)


# def plot_():
#
#     non_zero_indices = model.feature_importances_ >= 0.001
#     non_zero_cols = X_all.columns[non_zero_indices]
#     non_zero_importances = xgb_model.feature_importances_[non_zero_indices]
#     sorted_non_zero_indices = non_zero_importances.argsort()
#     plt.barh(non_zero_cols[sorted_non_zero_indices], non_zero_importances[sorted_non_zero_indices])
#     plt.xlabel("Xgboost Feature Importance")
#
#
# def plot_feature_importance_(X, y, model):
#     import matplotlib.pyplot as plt
#     all_cols = model.feature_importances_
#     non_zero_indices = model.feature_importances_ >= 0.001
#     non_zero_cols = X.columns[non_zero_indices]
#     non_zero_importances = model.feature_importances_[non_zero_indices]
#     sorted_non_zero_indices = non_zero_importances.argsort()
#     plt.barh(non_zero_cols[sorted_non_zero_indices], non_zero_importances[sorted_non_zero_indices])
#     plt.xlabel("Xgboost Feature Importance")


def oos_validate(targets_all, x_all, model, config):
    lon_lat = targets_all.positions
    weights = targets_all.weights
    observations = targets_all.observations
    predictions = predict.predict(x_all, model, interval=config.quantiles, lon_lat=lon_lat)
    if mpiops.chunk_index == 0:
        tags = model.get_predict_tags()
        y_true = targets_all.observations
        to_text = [predictions, y_true[:, np.newaxis], lon_lat]

        true_vs_pred = Path(config.output_dir).joinpath(config.name + "_oos_validation.csv")
        cols = tags + ['y_true', 'lon', 'lat']
        np.savetxt(true_vs_pred, X=np.hstack(to_text), delimiter=',',
                   fmt='%.8e',
                   header=','.join(cols),
                   comments='')
        scores = regression_validation_scores(observations, predictions, weights, model)
        score_string = "OOS Validation Scores:\n"
        for metric, score in scores.items():
            score_string += "{}\t= {}\n".format(metric, score)

        geoio.output_json(scores, Path(config.output_dir).joinpath(config.name + "_oos_validation_scores.json"))
        log.info(score_string)
