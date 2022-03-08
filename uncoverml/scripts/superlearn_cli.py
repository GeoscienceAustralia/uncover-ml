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
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from revrand.metrics import smse
from vecstack import stacking


import uncoverml.config
import uncoverml.scripts
from uncoverml.validate import regression_validation_scores
from uncoverml.geoio import RasterioImageSource, ImageWriter, get_image_spec
from uncoverml.features import extract_subchunks

from uncoverml.krige import krig_dict
from uncoverml.models import modelmaps, apply_multiple_masked
from uncoverml.targets import Targets
from uncoverml.optimise.models import transformed_modelmaps
from xgboost import XGBRegressor


_logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


all_modelmaps = {**transformed_modelmaps, **modelmaps, **krig_dict}

def define_model(alg_yaml):
    alg_config = uncoverml.config.Config(alg_yaml)
    model = all_modelmaps[alg_config.algorithm](**alg_config.algorithm_args)
    return model


# print(dir(uncoverml.scripts.cli))

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
    learn_lst = _grp(s, 'learning', "'learning' block must be provided when superlearning.")

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
    y = df['y_true'].values
    X = df.drop(columns='y_true').values
    super_cv(X, y)
    s_model = super_train(X, y)
    for sm in model_lst:
        model_full_train(sm, X, y)

    X_all = get_Xs(learn_lst)
    y_lr = super_predict(s_model, X_all, model_lst[-1], alg_yaml)
    super_tif(model_lst[-1], alg_yaml, y_lr)
    # vecstack
    weights = np.ones_like(y)
    super_stack = v_stack(X, y, bmodel_lst, XGBRegressor, sample_weight=weights,
        seed=0, n_jobs=-1, learning_rate=0.1, n_estimators=30, max_depth=3)
    y_all = super_stack.predict(X_all)
    super_tif(model_lst[-1], alg_yaml, y_lr, tif_name = "VSuper")
    return


def super_cv(X, y, n_splits=5):
    """
    ...
    """

    kfold = KFold(n_splits=n_splits)
    meta_X = []
    meta_y = []
    fold_scores = []

    for train_ix, test_ix in kfold.split(X):
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        sm_fold = SuperLearn(LinearRegression)
        sm_fold.fit(train_X, train_y)
        y_pred = sm_fold.predict(test_X)
        meta_y.extend(y_pred)
        fold_scores.append(regression_validation_scores(test_y, y_pred.reshape(-1, 1), np.ones_like(test_y), sm_fold))
    _logger.info(f"SUPER CV smse: {smse(y, meta_y)}")
    _logger.info(f"SUPER CV r2: {r2_score(y, meta_y)}")
    scores_df = pd.DataFrame(fold_scores)
    _logger.info(f"SUPER CV scores:\n{scores_df}")
    _logger.info(f"SUPER CV score averages:\n{ {c: scores_df[c].mean() for c in scores_df.columns} }")
    plt.figure()
    plt.scatter(y, meta_y)
    plt.xlabel("True target")
    plt.ylabel("Super-learner predicted");
    plt.savefig("Super_cv_predicted.png")
    plt.close()


def super_train(X, y):
    """
    ...
    """

    s_model = SuperLearn(LinearRegression)
    s_model.fit(X, y)
    _logger.info(f"SUPER coeffs: {s_model.model.coef_}")
    _logger.info(f"SUPER intercept: {s_model.model.intercept_}")
    y_super = s_model.model.predict(X[:, :])
    print("SUPER smse:", smse(y, y_super))
    print("SUPER r2:", r2_score(y, y_super))
    plt.figure()
    plt.scatter(y, y_super)
    plt.xlabel("True target")
    plt.ylabel("Super-learner predicted")
    plt.savefig("Super_train_predicted.png")
    plt.close()
    return s_model


def get_Xs(learn_lst):
    """
    ...
    """

    arr_lst = []
    for alg in learn_lst:
        alg_outdir = f"./{alg['algorithm']}_out"
        tif = alg_outdir + f"/{alg['algorithm']}_prediction.tif"
        img = RasterioImageSource(tif)
        arr_lst.append(extract_subchunks(img, 0, 1, 0)[:, 0, 0, 0])

    X_all = np.column_stack(arr_lst)
    return X_all


def super_predict(s_model, X_all, alg_model, alg_yaml):
    """
    ...
    """

    # y_all = s_model.intercept_ + X_all.dot(s_model.coef_)
    y_all = s_model.predict(X_all)
    _logger.debug("Feature-target shapes: ", X_all.shape, y_all.shape)
    return y_all


def super_tif(alg_model, alg_yaml, y_all, tif_name="Super"):
    """
    ...
    """
    alg_config = uncoverml.config.Config(alg_yaml)
    img_shape, img_bbox, img_crs = get_image_spec(alg_model, alg_config)
    img_out = ImageWriter(img_shape, img_bbox, img_crs, tif_name, 1, "./", band_tags=["Prediction"])
    img_out.write(y_all.view(np.ma.masked_array).reshape(-1, 1), 0)


def _load_model(model_file):
    with open(model_file, 'rb') as f:
        return joblib.load(f)


class MetaLRLearn(LinearRegression):
    def __init__(self, **kwargs):
        super(MetaLRLearn, self).__init__(**kwargs)
    def fit(self, X, y):
        super(MetaLRLearn, self).fit(X, y)
    def predict(self, X):
        return super(MetaLRLearn, self).predict(X)
    def get_predict_tags(self):
        return ['Prediction']


class SuperLearn:
    def __init__(self, meta_learn_class):
        self.model = meta_learn_class()
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def get_predict_tags(self):
        return ['Prediction']


def model_full_train(sm, X, y):
    """
    ...
    """

    weights = np.ones_like(y)
    sm.fit(X, y, sample_weight = weights)
    y_super = sm.predict(X[:, :])
    try:
        print(f"Model: {sm}")
    except:
        print(f"Model: {dir(sm)}")
    try:
        print("smse:", smse(y, y_super))
    except:
        print("CAN'T DO smse!")
    try:
        print("r2:", r2_score(y, y_super))
    except:
        print("CAN'T DO r2!")


def v_stack(X, y, models, meta_learn_class, **kwargs):
    """
    using vecstack Functional API (stacking() function)
    """

    # Make train/test split
    # As usual in machine learning task we have X_train, y_train, and X_test
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
        # test_size = 0.2, random_state = 0)

    # Caution! All models and parameter values are just
    # demonstrational and shouldn't be considered as recommended.
    # Initialize 1-st level models.
    # Compute stacking features
    S_train, S_ = stacking(models, X, y, X, sample_weight=kwargs['sample_weight'],
        regression = True, metric = smse, n_folds = 5,
        shuffle = True, random_state = 0, verbose = 2)

    # Initialize 2-nd level model
    s_model = meta_learn_class(**kwargs)

    # Fit 2-nd level model
    s_model = s_model.fit(S_train, y)
    return s_model

