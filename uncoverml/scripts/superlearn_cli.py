#################################################
# Author: Ante Bilic                            #
# Since: 16/02/2022                             #
# Copyright: Geoscience Australia MTworkflow    #
# Version: N/A                                  #
# Maintainer: Ante Bilic                        #
# Email: Ante.Bilic@ga.gov.au                   #
# Version: N/A                                  #
#################################################

"""
Run the uncoverml pipeline for super-learning and prediction.
.. program-output:: uncoverml --help
"""

import logging
import joblib
from pathlib import Path
import os
import warnings

import click
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from revrand.metrics import smse
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from vecstack import stacking
from mlens.ensemble import SuperLearner

import uncoverml.config
import uncoverml.scripts
from uncoverml.validate import regression_validation_scores
from uncoverml.geoio import RasterioImageSource, ImageWriter, get_image_spec
from uncoverml.features import extract_subchunks

from uncoverml.krige import krig_dict
from uncoverml.models import modelmaps, apply_multiple_masked
from uncoverml.targets import Targets
from uncoverml.optimise.models import transformed_modelmaps

_logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

meta_map = {
    'XGBoost': XGBRegressor,
    'GradientBoost': GradientBoostingRegressor,
    'RandomForest': RandomForestRegressor,
    'Linear': LinearRegression
    }

all_modelmaps = {**transformed_modelmaps, **modelmaps, **krig_dict}


def define_model(alg_yaml):
    alg_config = uncoverml.config.Config(alg_yaml)
    model = all_modelmaps[alg_config.algorithm](**alg_config.algorithm_args)
    return model


def main(pipeline_file, partitions):
    """
    """

    def _grp(d, k, msg=None):
        """
        Get required parameter.
        """
        try:
            return d[k]
        except KeyError:
            if msg is None:
                msg = f"Required parameter {k} not present in config."
            _logger.exception(msg)
            raise

    # uncoverml.config.Config._configure_pyyaml()
    with open(pipeline_file, 'r') as f:
        try:
            s = yaml.safe_load(f)
        except UnicodeDecodeError:
            if pipeline_file.endswith('.model'):
                _logger.error("You're attempting to run uncoverml but have provided the "
                              "'.model' file instead of the '.yaml' config file. The predict "
                              "now requires the configuration file and not the model. Please "
                              "try rerunning the command with the configuration file.")
            else:
                _logger.error("Couldn't parse the yaml file. Ensure you've provided the correct "
                              "file as config file and that the YAML is valid.")

    if 'metalearning' in s:
        meta_alg = s['metalearning']['algorithm']
        meta_learner = meta_map.get(meta_alg)
        try:
            meta_args = s['metalearning']['arguments']
        except KeyError as _e:
            meta_args = {}
        if meta_args is None:
            # empty "arguments:"
            meta_args = {}
        _logger.info(f"Using meta-learner {meta_map.get(meta_alg)} with \n{meta_args}")
    else:
        meta_learner = LinearRegression
        meta_args = {}
        _logger.info("Using the default meta-learner LinearRegression")

    learn_lst = _grp(s, 'learning', "'learning' block must be provided when metalearning.")
    s.pop("learning")
    ddd = {}
    model_lst = []
    bmodel_lst = []
    df = pd.DataFrame()
    for alg in learn_lst:
        _logger.info(f"Base model {alg['algorithm']} learning about to begin...")
        ddd.update({"learning": alg})
        ddd.update(s)
        alg_outdir = f"./{alg['algorithm']}_out"
        ddd["output"]["directory"] = alg_outdir
        ddd["output"]["model"] = f"{alg['algorithm']}.model"
        ddd["pickling"]["covariates"] = alg_outdir + "/features.pk"
        ddd["pickling"]["targets"] = alg_outdir + "/targets.pk"
        alg_yaml = f"./{alg['algorithm']}.yml"
        with open(alg_yaml, 'w') as yout:
            yaml.dump(ddd, yout, default_flow_style=False, sort_keys=False)
        uncoverml.scripts.uncoverml.learn.callback(alg_yaml, [], partitions)

        model_conf = _load_model(ddd["output"]["directory"] + "/" + ddd["output"]["model"])
        model_lst.append(model_conf["model"])
        bmodel_lst.append(define_model(alg_yaml))

        retain = None
        mb = s.get('mask')
        if mb:
            mask = mb.get('file')
            if not os.path.exists(mask):
                raise FileNotFoundError("Mask file provided in config does not exist. Check that the 'file' property of the 'mask' block is correct.")
            retain = _grp(mb, 'retain', "'retain' must be provided if providing a prediction mask.")
        else:
            mask = None

        _logger.info(f"Base model {alg['algorithm']} predictions about to begin...")
        try:
            uncoverml.scripts.uncoverml.predict.callback(ddd["output"]["directory"] + "/" + ddd["output"]["model"], partitions, mask, retain)
        except TypeError:
            _logger.error(f"Learner {alg['algorithm']} cannot predict")

        pred = pd.read_csv(alg_outdir + "/" + f"{alg['algorithm']}_results.csv")
        pred.rename(columns={'y_pred': alg['algorithm']}, inplace=True)
        pred.drop(columns=['y_transformed'], inplace=True)
        if df.empty:
            df = pred
        else:
            df = df.merge(pred, on=['y_true', 'lat', 'lon'])

    df.drop(columns=['lat', 'lon'], inplace=True)
    _logger.info(f"\nBase model training set predictions:\n{df.head()}")
    _logger.info(f"...\n{df.tail()}")

    X_meta = get_Xs(learn_lst)

    column_nan = np.logical_or(np.isnan(X_meta).any(axis=0), np.isinf(X_meta).any(axis=0))
    if column_nan.any():
        _logger.info(f"Inspect learner rasters for NaNs: {*column_nan,}")
        colnan_ix = np.argwhere(column_nan)
        _logger.info(f"Problem column(s) with NaNs: {*colnan_ix,}")
        X_meta = np.delete(X_meta, colnan_ix[0], axis=1)
        for _i in np.argwhere(column_nan)[0][::-1]:
            learn_lst.pop(_i)
            model_lst.pop(_i)
            bmodel_lst.pop(_i)
        df.drop(df.columns[colnan_ix[0]], axis=1, inplace=True)

    y = df['y_true'].values
    X = df.drop(columns='y_true').values
    if 'validation' in s:
        try:
            cv_folds = int(s['validation'][1]['k-fold']['folds'])
        except (KeyError, TypeError) as _e:
            _logger.exception(f"CV fold unreadable {_e}")
            cv_folds = 5
        _logger.info(f"meta-learner CV with {cv_folds} folds")

    meta_cv(X, y, meta_learner, n_splits=cv_folds, **meta_args)
    meta_model = meta_train(X, y, meta_learner, **meta_args)
    for bm in model_lst:
        model_full_train(bm, X, y)

    y_lr = meta_predict(meta_model, X_meta)
    meta_tif(model_lst[-1], alg_yaml, y_lr)
    # vecstack
    weights = np.ones_like(y)
    meta_stack = v_stack(X, y, bmodel_lst, meta_learner, sample_weight=weights, **meta_args)
    y_meta = meta_stack.predict(X_meta)
    meta_tif(model_lst[-1], alg_yaml, y_meta, tif_name = "VSuper")
    # MLEns
    mle_stack = m_stack(X, y, bmodel_lst, meta_learner, **meta_args)
    y_mle = mle_stack.predict(X_meta)
    meta_tif(model_lst[-1], alg_yaml, y_mle, tif_name = "MSuper")
    return


def meta_cv(X, y, meta_learner, n_splits=5, **kwargs):
    """
    ...
    """

    kfold = KFold(n_splits=n_splits)
    y_pred = []
    fold_scores = []

    for train_ix, test_ix in kfold.split(X):
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        _sm = MetaLearn(meta_learner, **kwargs)
        _sm.fit(train_X, train_y)
        y_oof = _sm.predict(test_X)
        y_pred.extend(y_oof)
        fold_scores.append(regression_validation_scores(test_y, y_oof.reshape(-1, 1), np.ones_like(test_y), _sm))
    _logger.info(f"SUPER CV smse: {smse(y, y_pred)}")
    _logger.info(f"SUPER CV r2: {r2_score(y, y_pred)}")
    scores_df = pd.DataFrame(fold_scores)
    _logger.info(f"SUPER CV scores:\n{scores_df}")
    _logger.info(f"SUPER CV score averages:\n{ {c: scores_df[c].mean() for c in scores_df.columns} }")
    plt.figure()
    plt.scatter(y, y_pred)
    plt.xlabel("True target")
    plt.ylabel("Meta-learner predicted");
    plt.savefig("Meta_cv_predicted.png")
    plt.close()


def meta_train(X, y, meta_learner, **kwargs):
    """
    ...
    """

    meta_model = MetaLearn(meta_learner, **kwargs)
    meta_model.fit(X, y)
    if isinstance(meta_model, LinearRegression):
        _logger.info(f"SUPER coeffs: {meta_model.model.coef_}")
        _logger.info(f"SUPER intercept: {meta_model.model.intercept_}")
    y_pred = meta_model.model.predict(X[:, :])
    print("SUPER smse:", smse(y, y_pred))
    print("SUPER r2:", r2_score(y, y_pred))
    plt.figure()
    plt.scatter(y, y_pred)
    plt.xlabel("True target")
    plt.ylabel("Meta-learner predicted")
    plt.savefig("Meta_train_predicted.png")
    plt.close()
    return meta_model


def get_Xs(learn_lst):
    """
    ...
    """

    arr_lst = []
    for _i, alg in enumerate(learn_lst):
        alg_outdir = f"./{alg['algorithm']}_out"
        tif = alg_outdir + f"/{alg['algorithm']}_prediction.tif"
        img = RasterioImageSource(tif)
        marr = extract_subchunks(img, 0, 1, 0)
        arr_lst.append(marr[:, 0, 0, 0])

    X_meta = np.column_stack(arr_lst)
    return X_meta


def meta_predict(meta_model, X_meta):
    """
    ...
    """

    y_meta = meta_model.predict(X_meta)
    _logger.debug("Feature-target shapes: ", X_meta.shape, y_meta.shape)
    return y_meta


def meta_tif(alg_model, alg_yaml, y_meta, tif_name="Meta"):
    """
    ...
    """
    alg_config = uncoverml.config.Config(alg_yaml)
    img_shape, img_bbox, img_crs = get_image_spec(alg_model, alg_config)
    img_out = ImageWriter(img_shape, img_bbox, img_crs, tif_name, 1, "./", band_tags=["Prediction"])
    img_out.write(y_meta.view(np.ma.masked_array).reshape(-1, 1), 0)


def _load_model(model_file):
    with open(model_file, 'rb') as f:
        return joblib.load(f)


def getMetaLearner(meta_learner):
    """
    Allows for the super-class as a variable 
    """

    class MetaLearnX(meta_learner):
        """
        ...
        """
        def __init__(self, **kwargs):
            super(MetaLearnX, self).__init__(**kwargs)
        def fit(self, X, y):
            super(MetaLearnX, self).fit(X, y)
        def predict(self, X):
            return super(MetaLearnX, self).predict(X)
        def get_predict_tags(self):
            return ['Prediction']

    return MetaLearnX


class MetaLearn:
    def __init__(self, meta_learner, **kwargs):
        self.model = meta_learner(**kwargs)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def get_predict_tags(self):
        return ['Prediction']


def model_full_train(_bm, X, y):
    """
    ...
    """

    weights = np.ones_like(y)
    _bm.fit(X, y, sample_weight = weights)
    y_pred = _bm.predict(X[:, :])
    try:
        print(f"Model: {_bm}")
    except:
        print(f"Model: {dir(_bm)}")
    try:
        print("smse:", smse(y, y_pred))
    except:
        print("CAN'T DO smse!")
    try:
        print("r2:", r2_score(y, y_pred))
    except:
        print("CAN'T DO r2!")


def v_stack(X, y, models, meta_learner, **kwargs):
    """
    using vecstack Functional API (stacking() function)
    """

    sample_weight = kwargs.get("sample_weight")
    # Compute stacking features
    if sample_weight is None:
        S_train, S_ = stacking(models, X, y, X,
            regression=True, metric=smse, n_folds=5,
            shuffle=True, random_state=0, verbose=2)
    else:
        S_train, S_ = stacking(models, X, y, X, sample_weight=sample_weight,
            regression=True, metric=smse, n_folds=5,
            shuffle=True, random_state=0, verbose=2)
        kwargs.pop("sample_weight")

    # Initialize 2-nd level model
    meta_model = meta_learner(**kwargs)

    # Fit 2-nd level model
    meta_model = meta_model.fit(S_train, y)
    return meta_model


def m_stack(X, y, models, meta_learner, **kwargs):
    """
    using vecstack Functional API (stacking() function)
    """

    ensemble = SuperLearner(scorer=smse, folds=5, shuffle=True, sample_size=X.shape[0])
    ensemble.add(models)
    ensemble.add_meta(meta_learner(**kwargs))
    # Fit 2-nd level model
    ensemble.fit(X, y)
    return ensemble

