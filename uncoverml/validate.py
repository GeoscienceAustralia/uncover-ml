""" Scripts for validation """

from __future__ import division
import logging
import copy
import json
import os
import pickle

from pathlib import Path
import numpy as np
from sklearn.metrics import (explained_variance_score, r2_score,
                             accuracy_score, log_loss, roc_auc_score,
                             confusion_matrix)
import eli5
from eli5.sklearn import PermutationImportance
from revrand.metrics import lins_ccc, mll, smse
import pandas as pd
import matplotlib.pyplot as plt

from uncoverml.models import apply_multiple_masked
from uncoverml import mpiops, diagnostics
from uncoverml import predict, geoio
from uncoverml import features as feat
from uncoverml import targets as targ
from uncoverml.learn import all_modelmaps as modelmaps
from uncoverml.optimise.models import transformed_modelmaps


_logger = logging.getLogger(__name__)

MINPROB = 1e-5  # Numerical guard for log-loss evaluation

regression_metrics = {
    'r2_score': lambda y, py, vy, y_t, py_t, vy_t:  r2_score(y, py),
    'expvar': lambda y, py, vy, y_t, py_t, vy_t: explained_variance_score(y, py),
    'smse': lambda y, py, vy, y_t, py_t, vy_t: smse(y, py),
    'lins_ccc': lambda y, py, vy, y_t, py_t, vy_t: lins_ccc(y, py),
    'mll': lambda y, py, vy, y_t, py_t, vy_t: mll(y, py, vy)
}

transformed_regression_metrics = {
    'r2_score_transformed': lambda y, py, vy, y_t, py_t, vy_t: r2_score(y_t, py_t),
    'expvar_transformed': lambda y, py, vy, y_t, py_t, vy_t: explained_variance_score(y_t, py_t),
    'smse_transformed': lambda y, py, vy, y_t, py_t, vy_t: smse(y_t, py_t),
    'lins_ccc_transformed': lambda y, py, vy, y_t, py_t, vy_t: lins_ccc(y_t, py_t),
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
    'mean_confusion': lambda y, ey, p: (confusion_matrix(y, ey)).tolist(),
    'mean_confusion_normalized': lambda y, ey, p:
        (confusion_matrix(y, ey) / len(y)).tolist()
}

def adjusted_r2_score(r2, n_samples, n_covariates):
    return 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_covariates - 1))

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


def regression_validation_scores(y, ey, n_covariates, model):
    """ Calculates the validation scores for a regression prediction
    Given the test and training data, as well as the outputs from every model,
    this function calculates all of the applicable metrics in the following
    list, and returns a dictionary with the following (possible) keys:
        + r2_score
        + expvar
        + smse
        + lins_ccc
        + mll

    Parameters
    ----------
    y: numpy.array
        The test data outputs
    ey: numpy.array
        The predictions made by the trained model on test data
    n_covariates: int
        The number of covariates being used.

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

    scores['adjusted_r2_score'] = adjusted_r2_score(scores['r2_score'], y.shape[0], n_covariates)
    if 'r2_score_transformed' in scores:
        scores['adjusted_r2_score_transformed'] = \
            adjusted_r2_score(scores['r2_score_transformed'], y_t.shape[0], n_covariates)

    return scores


class CrossvalInfo:
    def __init__(self, scores, y_true, y_pred, classification, positions):
        self.scores = scores
        self.y_true = y_true
        self.y_pred = y_pred
        self.classification = classification
        self.positions = positions

    def export_crossval(self, config):
        """
        Exports a CSV file containing real target values and their
        corresponding predicted value generated as part of 
        cross-validation. 

        Also populates the 'prediction' column of the 'rawcovariates'
        CSV file. 

        If enabled, the real vs predicted values will be plotted.

        Parameters
        ----------
        config: Config
            Uncover-ml config object.
        """
        # Make sure we convert numpy arrays to lists
        scores = {s: v if np.isscalar(v) else v.tolist()
                  for s, v in self.scores.items()}

        with open(config.crossval_scores_file, 'w') as f:
            json.dump(scores, f, sort_keys=True, indent=4)

        to_text = [self.y_true, self.y_pred['Prediction'], self.positions[:,0], self.positions[:,1]]

        np.savetxt(config.crossval_results_file, X=np.vstack(to_text).T, 
                   delimiter=',', fmt='%.4f', header='y_true,y_pred,x,y')

        if os.path.exists(config.raw_covariates) and os.path.exists(config.raw_covariates_mask):
            # Also add prediction values to rawcovariates.csv - yes this file 
            # is very overloaded and we need to fix the output situation.
            # Get indicies sorted by location so we can insert the 
            # prediction in the correct row.
            inds = np.lexsort(self.positions.T)
            sorted_predictions = self.y_pred['Prediction'][inds]

            # If null rows have been dropped, we need to add these to 
            # the prediction array - otherwise we'll get a length 
            # mismatch error between number of predictions and number 
            # of rows in the table.
            masked_rcv = pd.read_csv(config.raw_covariates_mask, delimiter=',')
            mask = masked_rcv.apply(np.sum, axis=1).to_numpy().astype(bool)
            masked_predictions = np.zeros(mask.shape)

            pred_iter = np.nditer(sorted_predictions)
            mask_iter = np.nditer(mask, flags=['c_index'])
            with pred_iter, mask_iter:
                while not mask_iter.finished:
                    if mask_iter[0]:
                        masked_predictions[mask_iter.index] = pred_iter[0]
                        pred_iter.iternext()
                    mask_iter.iternext()

            rcv = pd.read_csv(config.raw_covariates, delimiter=',')
            rcv['prediction'] = masked_predictions
            rcv.to_csv(config.raw_covariates, sep=',')

            if config.plot_real_vs_pred:
                diagnostics.plot_real_vs_pred_crossval(
                    config.crossval_results_file,
                    scores_path=config.crossval_scores_file,
                    bins=40, overlay=True,
                    hist_cm=plt.cm.Oranges, scatter_color='black'
                ).savefig(config.plot_real_vs_pred)
                diagnostics.plot_residual_error_crossval(
                    config.crossval_results_file
                ).savefig(config.plot_residual)

        else:
            _logger.warning("Cross-validation results are being exported but rawcovariates.csv "
                            "and/or raw_covaraites_mask.csv do not exist in output directory. "
                            "Cross-validation predictions won't be added to rawcovariates table "
                            "and 'Real vs Pred' will not be plotted. To resolve this, re-run "
                            "learn without using pickled covariate and target data.")


class OOSInfo(CrossvalInfo):
    def export_scores(self, config):
        scores = {s: v if np.isscalar(v) else v.tolist()
                  for s, v in self.scores.items()}

        with open(config.oos_scores_file, 'w') as f:
            json.dump(scores, f, sort_keys=True, indent=4)

        to_text = [self.y_true, self.y_pred['Prediction'], self.positions[:,0], self.positions[:,1]]

        np.savetxt(config.oos_results_file, X=np.vstack(to_text).T, 
                   delimiter=',', fmt='%.4f', header='y_true,y_pred,x,y')


def permutation_importance(model, x_all, targets_all, config):
    _logger.info("Computing permutation importance!!")
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
                                      refit=False).fit, data=(x_all, y)
            )
            feature_names = geoio.feature_names(config)
            df_picv = eli5.explain_weights_df(
                pi_cv, feature_names=feature_names, top=100)
            csv = Path(config.output_dir).joinpath(
                config.name + "_permutation_importance_{}.csv".format(
                    score)).as_posix()
            df_picv.to_csv(csv, index=False)

def out_of_sample_validation(model, targets, features, config):
    _logger.info(
        f"Performing out-of-sample validation with {targets.observations.shape[0]} targets...")
    mpiops.comm_world.barrier()
    if mpiops.leader_world:
        with open(config.model_file, 'rb') as f:
            model, _, _ = pickle.load(f)
    model = mpiops.comm_world.bcast(model, root=0)
    classification = hasattr(model, 'predict_proba')
    pos = np.array_split(targets.positions, mpiops.size_world)[mpiops.rank_world]
    fields = {}
    for k, v in targets.fields.items():
        fields[k] = np.array_split(v, mpiops.size_world)[mpiops.rank_world]
    features = np.array_split(features, mpiops.size_world)[mpiops.rank_world]
    pred = predict.predict(features, model,
                           fields=fields,
                           lon_lat=pos)

    pred = mpiops.comm_world.gather(pred, root=0)
    if mpiops.leader_world:
        pred = np.concatenate(pred)
        if classification:
            hard, p = pred[:, 0], pred[:, 1:]
            scores = classification_validation_scores(targets.observations, hard, p)
        else:
            scores = regression_validation_scores(targets.observations, pred, features.shape[1], model)

        _logger.info("Out of sample validation complete, scores:")
        for k, v in scores.items():
            _logger.info(f"{k}: {v}")

        result_tags = model.get_predict_tags()
        y_pred_dict = dict(zip(result_tags, pred.T))
        if hasattr(model, '_notransform_predict'):
            y_pred_dict['transformedpredict'] = \
                model.target_transform.transform(pred[:, 0])
        
        return OOSInfo(scores, targets.observations, y_pred_dict, classification, targets.positions)
    else:
        return None

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
        _logger.info("Computing {} feature importance of {}"
                 .format(config.algorithm, fname))
        x, keep = feat.transform_features(image_chunks_leaveout,
                                          transform_sets_leaveout,
                                          final_transform_leaveout,
                                          config)
        x_all = feat.gather_features(x[keep], node=0)
        targets_all = targ.gather_targets_main(targets, keep, node=0)


        # Feature ranking occurs before top-level shared training data
        # is created, so share the memory now so we can parallel 
        # validate.
        if config.parallel_validate:
            # Send targets/covariates to local leaders so they can share in their respective nodes
            if mpiops.leader_world:
                for v in mpiops.node_map.values():
                    if v != 0:
                        mpiops.comm_world.send(targets_all, dest=v, tag=v + 1)
                        mpiops.comm_world.send(x_all, dest=v, tag=v + 2)
            elif mpiops.leader_local:
                targets_all = ls.mpiops.comm_world.recv(
                        source=mpiops.leader_world, tag=mpiops.rank_world + 1)
                x_all = mpiops.comm_world.recv(
                        source=ls.mpiops.leader_world, tag=mpiops.rank_world + 2)
                mpiops.comm_world.barrier()

            training_data = geoio.create_shared_training_data(targets_all, x_all)
            targets_all = training_data.targets_all
            x_all = training_data.x_all
        
        results = local_crossval(x_all, targets_all, config)
        feature_scores[fname] = results

        geoio.deallocate_shared_training_data(training_data)
    
    # Get the different types of score from one of the outputs
    if mpiops.leader_world:
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
    parallel_model = config.multicubist or config.multirandomforest or config.bootstrap
    if config.bootstrap and config.parallel_validate:
        config.alrgorithm_args['parallel'] = False
    elif not config.bootstrap and not config.parallel_validate and not mpiops.leader_world:
        return

    if config.multicubist or config.multirandomforest:
        config.algorithm_args['parallel'] = False
   
    _logger.info("Validating with {} folds".format(config.folds))
    model = modelmaps[config.algorithm](**config.algorithm_args)
    classification = hasattr(model, 'predict_proba')
    y = targets_all.observations
    lon_lat = targets_all.positions
    _, cv_indices = split_cfold(y.shape[0], config.folds, config.crossval_seed)

    # Split folds over workers
    fold_list = np.arange(config.folds)
    if config.parallel_validate:
        fold_node = \
            np.array_split(fold_list, mpiops.size_world)[mpiops.rank_world]
    else:
        fold_node = fold_list

    y_pred = {}
    y_true = {}
    fold_scores = {}
    pos = {}

    # Train and score on each fold
    for fold in fold_node:
        _logger.info(":mpi:Training fold {} of {}".format(
            fold + 1, config.folds, mpiops.rank_world))

        train_mask = cv_indices != fold
        test_mask = ~ train_mask
    
        y_k_train = y[train_mask]
        if config.target_weight_property:
            y_k_weight = targets_all.fields[config.target_weight_property][train_mask]
        else:
            y_k_weight = None
        lon_lat_train = lon_lat[train_mask]
        lon_lat_test = lon_lat[test_mask]

        # Extra fields
        fields_train = {f: v[train_mask]
                        for f, v in targets_all.fields.items()}
        fields_pred = {f: v[test_mask] for f, v in targets_all.fields.items()}

        # Train on this fold
        x_train = x_all[train_mask]
        apply_multiple_masked(model.fit, data=(x_train, y_k_train), fields=fields_train,
                              lon_lat=lon_lat_train, 
                              sample_weight=y_k_weight)

        # Testing
        if not config.parallel_validate and not mpiops.leader_world:
            continue
        else:
            y_k_pred = predict.predict(x_all[test_mask], model,
                                       fields=fields_pred,
                                       lon_lat=lon_lat_test)
            y_pred[fold] = y_k_pred
            n_covariates = x_all[test_mask].shape[1]

            # Regression
            if not classification:
                y_k_test = y[test_mask]
                fold_scores[fold] = regression_validation_scores(
                    y_k_test, y_k_pred, n_covariates, model)

            # Classification
            else:
                y_k_test = model.le.transform(y[test_mask])
                y_k_hard, p_k = y_k_pred[:, 0], y_k_pred[:, 1:]
                fold_scores[fold] = classification_validation_scores(
                    y_k_test, y_k_hard, p_k
                )
            
            y_true[fold] = y_k_test
            pos[fold] = lon_lat_test

    if config.parallel_validate:
        y_pred = _join_dicts(mpiops.comm_world.gather(y_pred, root=0))
        y_true = _join_dicts(mpiops.comm_world.gather(y_true, root=0))
        pos = _join_dicts(mpiops.comm_world.gather(pos, root=0))
        scores = _join_dicts(mpiops.comm_world.gather(fold_scores, root=0))
    else:
        scores = fold_scores

    result = None
    if mpiops.leader_world:
        y_true = np.concatenate([y_true[i] for i in range(config.folds)])
        y_pred = np.concatenate([y_pred[i] for i in range(config.folds)])
        pos = np.concatenate([pos[i] for i in range(config.folds)])
        valid_metrics = scores[0].keys()
        scores = {m: np.mean([d[m] for d in scores.values()], axis=0)
                  for m in valid_metrics}
        score_string = "Validation complete:\n"
        for metric, score in scores.items():
            score_string += "{}\t= {}\n".format(metric, score)
        _logger.info(score_string)

        result_tags = model.get_predict_tags()
        y_pred_dict = dict(zip(result_tags, y_pred.T))
        if hasattr(model, '_notransform_predict'):
            y_pred_dict['transformedpredict'] = \
                model.target_transform.transform(y_pred[:, 0])
        result = CrossvalInfo(scores, y_true, y_pred_dict, classification, pos)

    if parallel_model:
        config.algorithm_args['parallel'] = True

    return result
