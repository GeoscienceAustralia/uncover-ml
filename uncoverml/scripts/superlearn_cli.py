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

import uncoverml.config
import uncoverml.scripts
from uncoverml.geoio import RasterioImageSource, ImageWriter, get_image_spec
from uncoverml.features import extract_subchunks

_logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


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
    super_predict(s_model, learn_lst, model_lst[-1], alg_yaml)
    return


def super_cv(X, y, n_splits=5):
    """
    ...
    """

    kfold = KFold(n_splits=n_splits)
    meta_X = []
    meta_y = []

    for train_ix, test_ix in kfold.split(X):
        y_fold = []
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        sm_fold = fit_super_model(train_X, train_y)
        meta_y.extend(sm_fold.predict(test_X))
    _logger.info(f"SUPER CV smse: {smse(y, meta_y)}")
    _logger.info(f"SUPER CV r2: {r2_score(y, meta_y)}")
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

    s_model = fit_super_model(X, y)
    _logger.info(f"SUPER coeffs: {s_model.coef_}")
    _logger.info(f"SUPER intercept: {s_model.intercept_}")
    y_super = s_model.predict(X[:, :])
    print("SUPER smse:", smse(y, y_super))
    print("SUPER r2:", r2_score(y, y_super))
    plt.figure()
    plt.scatter(y, y_super)
    plt.xlabel("True target")
    plt.ylabel("Super-learner predicted")
    plt.savefig("Super_train_predicted.png")
    plt.close()
    return s_model


def super_predict(s_model, learn_lst, alg_model, alg_yaml):
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
    y_all = s_model.intercept_ + X_all.dot(s_model.coef_)
    _logger.debug("Feature-target shapes: ", X_all.shape, y_all.shape)

    alg_config = uncoverml.config.Config(alg_yaml)
    img_shape, img_bbox, img_crs = get_image_spec(alg_model, alg_config)
    img_out = ImageWriter(img_shape, img_bbox, img_crs, "Super", 1, "./", band_tags=["Prediction"])
    img_out.write(y_all.reshape(-1, 1), 0)


def _load_model(model_file):
    with open(model_file, 'rb') as f:
        return joblib.load(f)


def fit_super_model(X, y):
    s_model = LinearRegression()
    s_model.fit(X, y)
    return s_model
