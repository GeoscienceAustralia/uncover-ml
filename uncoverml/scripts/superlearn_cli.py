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
Runs uncoverml for super-learning and prediction.
.. program-output:: uncoverml --help
"""

import logging
import joblib
from pathlib import Path
import os
import warnings
from typing import List, Tuple, Dict, Optional, Union, Any

import click
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from revrand.metrics import smse, lins_ccc
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.base import RegressorMixin
from vecstack import stacking, StackingTransformer
from mlens.ensemble import SuperLearner

from uncoverml.config import Config
import uncoverml.scripts
from uncoverml.validate import regression_validation_scores
from uncoverml.geoio import RasterioImageSource, ImageWriter, get_image_spec
from uncoverml.features import extract_subchunks
from uncoverml.predict import _get_data

from uncoverml.krige import krig_dict
from uncoverml.models import modelmaps, apply_multiple_masked, TagsMixin
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


def define_model(alg_yaml: str) -> TagsMixin:
    """
    Returns the base-learner model from the yaml file and mapping above

    Parameters
    ----------
    dalg_yaml: str
        The YAML file in the current foder named after the learner algorithm
        with the associated keyword arguments.

    Returns
    ----------
    model: TagsMixin
        The base-learner model defined with the kewyord arguments
    """

    alg_config = Config(alg_yaml)
    model = all_modelmaps[alg_config.algorithm](**alg_config.algorithm_args)
    return model


def load_model(model_file: str) -> Dict[str, Union[TagsMixin, Config]]:
    """
    Returns a trained base-learner and its config paramaters as a dict
    {"model", base-learner model, "config": Config) from the *.model file

    Parameters
    ----------
    model_file: str
        The *.model file

    Returns
    ----------
        : Dict[model, config]
        Tre trained base-learner and its config parameters
    """

    with open(model_file, 'rb') as f:
        return joblib.load(f)


def main(pipeline_file: str, partitions: int) -> None:
    """
    The super-learning driver routine.

    Parameters
    ----------
    pipeline_file: str
        The YAML config file input script with the definitions of meta-learner,
        base-learners with the associated keyword arguments, covariates, targets,
        validation, prediction etc...
    partitions: int
        The number of data partitions (supplied as the command-line argument
        to uncoverml with the options -p)
    """

    def _grp(dic: Dict[str, str], _k: str, msg=None) -> str:
        """
        A ridiculous helper routine to read the keys k from a config dictionary.
        (taken from the develop branch config reader)

        Parameters
        ----------
        dic: Dict
            The dictionary obtained from the confing YAML file in the current folder.

        Returns
        ----------
        model: TagsMixin
            The base-learner model defined with the kewyord arguments
        """

        try:
            return dic[_k]
        except KeyError:
            if msg is None:
                msg = f"Required parameter {_k} not present in config."
            _logger.exception(msg)
            raise

    with open(pipeline_file, 'r') as f:
        try:
            conf_params = yaml.safe_load(f)
        except UnicodeDecodeError:
            if pipeline_file.endswith('.model'):
                _logger.error("You're attempting to run uncoverml but have provided the "
                              "'.model' file instead of the '.yaml' config file. The predict "
                              "now requires the configuration file and not the model. Please "
                              "try rerunning the command with the configuration file.")
            else:
                _logger.error("Couldn't parse the yaml file. Ensure you've provided the correct "
                              "file as config file and that the YAML is valid.")

    if 'metalearning' in conf_params:
        meta_alg = conf_params['metalearning']['algorithm']
        meta_learner = meta_map.get(meta_alg)
        try:
            meta_args = conf_params['metalearning']['arguments']
        except KeyError as _e:
            meta_args = {}
        if meta_args is None:
            # empty "arguments:" block
            meta_args = {}
        _logger.info(f"Using meta-learner {meta_map.get(meta_alg)} with \n{meta_args}")
    else:
        meta_learner = LinearRegression
        meta_args = {}
        _logger.info("Using the default meta-learner LinearRegression")

    learn_alg_lst = _grp(conf_params, 'learning', "'learning' block must be provided when metalearning.")
    conf_params.pop("learning")
    # the list of base-learner models (each as an object initialised with class(parameters)):
    base_learner_lst = []
    # the list of trained base-learner models (each as a dict {"model": ... , "config": ...} binary values):
    model_lst = []
    df = pd.DataFrame()

    for alg in learn_alg_lst:
        _logger.info(f"\nBase model {alg['algorithm']} learning about to begin...")
        learner_dic = {"learning": alg}
        learner_dic.update(conf_params)
        try:
            learner_dic.pop("metalearning")
        except KeyError:
            # if no "metalearning", LinearRegression() used as the super-learner
            pass
        alg_outdir = f"./{alg['algorithm']}_out"
        learner_dic["output"]["directory"] = alg_outdir
        learner_dic["output"]["model"] = f"{alg['algorithm']}.model"
        learner_dic["pickling"]["covariates"] = alg_outdir + "/features.pk"
        learner_dic["pickling"]["targets"] = alg_outdir + "/targets.pk"
        alg_yaml = f"./{alg['algorithm']}.yml"
        with open(alg_yaml, 'w') as yout:
            yaml.dump(learner_dic, yout, default_flow_style=False, sort_keys=False)
        uncoverml.scripts.uncoverml.learn.callback(alg_yaml, [], partitions)

        # define the base model with the YAML file parameters & append it to the list
        base_learner_lst.append(define_model(alg_yaml))
        # load the trained base model with the *.model file & append it to the model list
        model_conf = load_model(learner_dic["output"]["directory"] + "/" + learner_dic["output"]["model"])
        model_lst.append(model_conf)

        retain = None
        mb = conf_params.get('mask')
        if mb:
            mask = mb.get('file')
            if not os.path.exists(mask):
                raise FileNotFoundError("Mask file provided in config does not exist.")
            retain = _grp(mb, 'retain', "'retain' must be provided if providing a prediction mask.")
        else:
            mask = None

        _logger.info(f"Base model {alg['algorithm']} predictions about to begin...")
        txt = learner_dic["output"]["directory"] + "/" + learner_dic["output"]["model"]
        try:
            uncoverml.scripts.uncoverml.predict.callback(txt, partitions, mask, retain)
        except TypeError:
            _logger.error(f"Learner {alg['algorithm']} cannot predict")

        pred = pd.read_csv(alg_outdir + "/" + f"{alg['algorithm']}_results.csv")
        pred.rename(columns={'y_pred': alg['algorithm']}, inplace=True)
        pred.drop(columns=['y_transformed'], inplace=True)
        if df.empty:
            df = pred
        else:
            # due to shuffle=True the rows must be matched on y_true, lat & lon
            df = df.merge(pred, on=['y_true', 'lat', 'lon'])

    df.drop(columns=['lat', 'lon'], inplace=True)
    _logger.info(f"\nBase model training set predictions:\n{df.head()}")
    _logger.info(f"...\n{df.tail()}")

    base_alg_lst = [alg["algorithm"] for alg in learn_alg_lst]
    X_tif = get_metafeats(base_alg_lst)

    column_nan = np.logical_or(np.isnan(X_tif).any(axis=0), np.isinf(X_tif).any(axis=0))
    if column_nan.any():
        _logger.info(f"Inspect learner rasters for NaNs: {*column_nan,}")
        colnan_ix = np.argwhere(column_nan)
        _logger.info(f"Problem column(s) with NaNs: {*colnan_ix,}")
        X_tif = np.delete(X_tif, colnan_ix[0], axis=1)
        for _i in np.argwhere(column_nan)[0][::-1]:
            learn_alg_lst.pop(_i)
            model_lst.pop(_i)
            base_learner_lst.pop(_i)
        df.drop(df.columns[colnan_ix[0]], axis=1, inplace=True)

    y_mtrain = df['y_true'].values
    X_mtrain = df.drop(columns='y_true').values
    if 'validation' in conf_params:
        try:
            cv_folds = int(conf_params['validation'][1]['k-fold']['folds'])
        except (KeyError, TypeError) as _e:
            _logger.exception(f"CV fold unreadable {_e}")
            cv_folds = 5
        _logger.info(f"meta-learner CV with {cv_folds} folds")

    # meta-learn
    meta_cv(X_mtrain, y_mtrain, meta_learner, n_splits=cv_folds, **meta_args)
    meta_model = meta_train(X_mtrain, y_mtrain, meta_learner, **meta_args)
    # meta-predict
    y_meta_tif = meta_predict(meta_model, X_tif)
    target_tif = meta_tif(model_lst[-1], "Meta", 1, 0)
    target_tif.write(y_meta_tif.view(np.ma.masked_array).reshape(-1, 1), 0)
    target_tif.close()

    # load ALL training features & targets
    targets_shp, X_shp = uncoverml.scripts.uncoverml._load_data(model_lst[-1]["config"], partitions)
    # input(f"FIELDS: {targets_shp.fields}")
    # input(f"GROUPS: {targets_shp.groups}")
    # input(f"LATLON: {targets_shp.positions}")
    # input(f"WEIGHTS: {targets_shp.weights}")
    y_shp = targets_shp.observations
    weights = targets_shp.weights
    baselearn_tup = [(alg['algorithm'], _bl) for alg, _bl in zip(learn_alg_lst, base_learner_lst)]
    # vecstack learn
    stack, meta_model = skstack(X_shp, y_shp, baselearn_tup, meta_learner, sample_weight=weights, **meta_args)
    # MLEns learn
    mle_stack = m_stack(X_shp, y_shp, base_learner_lst, meta_learner, **meta_args)
    for _i in range(partitions):
        X_all, feature_names = _get_data(_i, model_lst[-1]["config"])
        if _i == 0:
            # initialize ImageWriter objects:
            starget_tif = meta_tif(model_lst[-1], "VSuper", partitions, _i)
            mtarget_tif = meta_tif(model_lst[-1], "MSuper", partitions, _i)
        # vecstack predict
        S_all = stack.transform(X_all)
        y_vstack_tif = meta_model.predict(S_all)
        starget_tif.write(y_vstack_tif.view(np.ma.masked_array).reshape(-1, 1), _i)
        # MLEns predict
        y_mstack_tif = mle_stack.predict(X_all)
        mtarget_tif.write(y_mstack_tif.view(np.ma.masked_array).reshape(-1, 1), _i)
    starget_tif.close()
    mtarget_tif.close()
    return


def read_tifs(tif_lst: List[str]) -> np.ma.MaskedArray:
    """
    Reads the features from the input geotif files and combines those into a 2D array
    as the super-learner predictor features (a 2D masked darray).

    Parameters
    ----------
    tif_lst: List[str]
        The names of the geotif files

    Returns
    ----------
    X_all: 2D MaskedArray
        The final predictor features for the super-learner
    """

    arr_lst = []
    for tif in tif_lst:
        img = RasterioImageSource(tif)
        marr = extract_subchunks(img, 0, 1, 0)
        arr_lst.append(marr[:, 0, 0, 0])

    X_all = np.column_stack(arr_lst)
    return X_all


def get_metafeats(base_alg_lst: List[str]) -> np.ma.MaskedArray:
    """
    Reads the predictions (geotifs) from the base-learners and combines those into a 2D array
    as the super-learner predictor features (a 2D masked darray).

    Parameters
    ----------
    base_alg_lst: List[str]
        The names of the base-learner algorithms

    Returns
    ----------
    X_tif: 2D MaskedArray
        The predictor features for the super-learner
    """

    arr_lst = []
    for alg in base_alg_lst:
        alg_outdir = f"./{alg}_out"
        tif = alg_outdir + f"/{alg}_prediction.tif"
        img = RasterioImageSource(tif)
        marr = extract_subchunks(img, 0, 1, 0)
        arr_lst.append(marr[:, 0, 0, 0])

    X_tif = np.column_stack(arr_lst)
    return X_tif


def meta_cv(X: np.ma.MaskedArray,
            y: np.ma.MaskedArray,
            meta_learner: RegressorMixin,
            n_splits: int=5,
            **kwargs: Union[Any, None]) -> None:
    """
    Runs the cross validation for the super-learner.

    Parameters
    ----------
    X: 2D array
        The training set feature values are the predictions of the base-learners
        over the samples obtained from the intersection of shapefile with covariate TIFs.
    y: 1D array
        The training set target values from the shapefile.
    meta_learner: one of the 4 Regressors (LinReg, XGBReg, GBM, RF)
        The super-learner class`
    n_splits: int
        The number of folds for the cross-validation
    **kwargs: any
        The optional keyword parameters for meta-learner initialisation
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
    y_pred = np.array(y_pred)
    _logger.info(f"SUPER CV r2 = {r2_score(y, y_pred)}")
    _logger.info(f"SUPER CV expvar = {explained_variance_score(y, y_pred)}")
    _logger.info(f"SUPER CV smse = {smse(y, y_pred)}")
    _logger.info(f"SUPER CV lins_ccc = {lins_ccc(y, y_pred)}")
    _logger.info(f"SUPER CV mse = {mean_squared_error(y, y_pred)}")
    scores_df = pd.DataFrame(fold_scores)
    _logger.info(f"SUPER CV scores:\n{scores_df}")
    _logger.info("SUPER CV score averages:")
    for _c in scores_df.columns:
        _logger.info(f"{_c} = {scores_df[_c].mean()}")
    plt.figure()
    plt.scatter(y, y_pred)
    plt.xlabel("True target")
    plt.ylabel("Meta-learner predicted");
    plt.savefig("Meta_cv_predicted.png")
    plt.close()


def meta_train(X: np.ma.MaskedArray,
                y: np.ma.MaskedArray,
                meta_learner: RegressorMixin,
                **kwargs: Union[Any, None]) -> RegressorMixin:
    """
    Trains the super-learner on the full training data set.

    Parameters
    ----------
    X: 2D array
        The training set feature values are the predictions of the base-learners
        over the samples obtained from the intersection of shapefile with covariate TIFs.
    y: 1D array
        The training set target values from the shapefile.
    meta_learner: RegressorMixin
         The super-learner class, one of the 4 Regressors (LinReg, XGBReg, GBM, RF)
    **kwargs: any
        The optional keyword parameters for meta-learner initialisation

    Returns
    ----------
    model: RegressorMixin
        The trained super-learner model.
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


def meta_predict(meta_model: RegressorMixin, X_tif: np.ma.MaskedArray) -> np.ndarray:
    """
    Makes predictions using the trained super-learner and the predictions from the
    base-learners as the features.

    Parameters
    ----------
    model: RegressorMixin
        The trained super-learner model.
    X_tif: 2D MaskedArray
        The predictor features for the super-learner with columns consisting of
        the predictions of base-learners and rows covering the area represented
        by the geotifs.

    Returns
    ----------
    y_tif: 1D array
        The trained super-learner model final predictions.
    """

    y_tif = meta_model.predict(X_tif)
    _logger.debug("Feature-target shapes: ", X_tif.shape, y_tif.shape)
    return y_tif


def meta_tif(model_config: Dict[str, Union[TagsMixin, Config]],
            tif_name: str="Meta",
            parts: int=1,
            idx: int=0) -> ImageWriter:
    """
    Converts the super-learner predictions into a geo-tif file.

    Parameters
    ----------
    model_config: Dict
        any trained base-learner and its config
    tif_name: str
        The geotif prefix name
    parts: int
        The number of the partition chunks
    idx: int
        The index of the partition chunk

    Returns
    ----------
    img_out: ImageWriter
        The image object to close when all partitions chunks are written
    """

    img_shape, img_bbox, img_crs = get_image_spec(model_config["model"], model_config["config"])
    img_out = ImageWriter(img_shape, img_bbox, img_crs, tif_name, parts, "./", band_tags=["Prediction"])
    return img_out


class MetaLearn:
    """
    The super-learner wrapper class and its essential methods: fit, predict & get_predict_tags.

    Parameters
    ----------
    meta_learner: RegressorMixin
         The super-learner actual class, one of the 4 Regressors (LinReg, XGBReg, GBM, RF)
    **kwargs: any
        The optional keyword parameters for meta-learner initialisation
    """

    def __init__(self, meta_learner: RegressorMixin, **kwargs: Union[Any, None]) -> None:
        """
        Assigns the model attribute to the Regressor defined by the meta_learner class and keywords.
        """
        self.model = meta_learner(**kwargs)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the super-learner on the full training data set.

        Parameters
        ----------
        X: 2D array
            The training set feature values are the predictions of the base-learners
            over the samples obtained from the intersection of shapefile with covariate TIFs.
        y: 1D array
            The training set target values from the shapefile.
        """

        self.model.fit(X, y)


    def predict(self, X_tif: np.ma.MaskedArray) -> np.ndarray:
        """
        Makes predictions using the trained super-learner and the predictions from the
        base-learners as the features.

        Parameters
        ----------
        X_tif: 2D MaskedArray
            The predictor features for the super-learner

        Returns
        ----------
        : 1D array
            The training set target values from the shapefile.
        """

        return self.model.predict(X_tif)


    def get_predict_tags(self) -> List[str]:
        """
        Required for compatibility with TagsMixin class which is a superclass
        of all models (see uncoverml/models.py)
        Get the types of prediction outputs from this algorithm.

        Returns
        -------
        : list of strings with the types of outputs that can be returned by this
            algorithm. Only ['Prediction'] supported here.
        """

        return ['Prediction']



def get_meta_learner(meta_learner) -> RegressorMixin:
    """
    The wrapper function simply to allow for the super-class to be a VARIABLE.
    (an alternative to the class MetaLearn above, currently unused)
    """

    class MetaLearnX(meta_learner):
        """
        The super-learner wrapper class and its essential methods: fit, predict & get_predict_tags.

        Parameters
        ----------
        meta_learner: RegressorMixin
             The super-learner actual class, one of the 4 Regressors (LinReg, XGBReg, GBM, RF)
        **kwargs: any
            The optional keyword parameters for meta-learner initialisation
        """

        def __init__(self, **kwargs) -> None:
            """
            the model is a Regressor defined by the meta_learner class and keywords.
            """

            super(MetaLearnX, self).__init__(**kwargs)


        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            """
            Trains the super-learner on the full training data set.

            Parameters
            ----------
            X: 2D array
                The training set feature values are the predictions of the base-learners
                over the samples obtained from the intersection of shapefile with covariate TIFs.
            y: 1D array
                The training set target values from the shapefile.
            """

            super(MetaLearnX, self).fit(X, y)


        def predict(self, X_tif: np.ma.MaskedArray) -> np.ndarray:
            """
            Makes predictions using the trained super-learner and the predictions from the
            base-learners as the features.

            Parameters
            ----------
            X_tif: 2D MaskedArray
                The predictor features for the super-learner

            Returns
            ----------
            : 1D array
                The training set target values from the shapefile.
            """

            return super(MetaLearnX, self).predict(X_tif)


        def get_predict_tags(self) -> List[str]:
            """
            Required for compatibility with TagsMixin class which is a superclass
            of all models (see uncoverml/models.py)
            Get the types of prediction outputs from this algorithm.

            Returns
            -------
            : list of strings with the types of outputs that can be returned by this
                algorithm. Only ['Prediction'] supported here.
            """

            return ['Prediction']


    return MetaLearnX


def v_stack(X: np.ndarray,
            y: np.ndarray,
            X_all: np.ndarray,
            models: List[TagsMixin],
            meta_learner: np.ndarray,
            **kwargs: Union[Any, None]) -> np.ndarray:
    """
    Trains the super-learner on the full training data set.
    using vecstack Functional API (i.e., the stacking() function) and
    makes predictions using the predictions from the base-learners as the features.

    Parameters
    ----------
    X: 2D array
        The training set feature values of covariates (columns) for the samples (rows)
        obtained from the intersection of shapefile with covariate TIFs.
    y: 1D array
        The training set target values from the shapefile.
    X_all: 2D array
        The final set feature values of covariates (columns) for the samples (rows)
        obtained from covariate geotifs.
    models: List[TagsMixin]
         The list of base-learners
    meta_learner: RegressorMixin
         The super-learner class, one of the 4 Regressors (LinReg, XGBReg, GBM, RF)
    **kwargs: any
        The optional keyword parameters for meta-learner initialisation

    Returns
    ----------
    y_all: np.ndarray
        The full predictions of the target over the area represented by the geotifs
    """

    sample_weight = kwargs.get("sample_weight")
    # Compute stacking features
    if sample_weight is None:
        S_train, S_all = stacking(models, X, y, X_all,
            regression=True, metric=smse, n_folds=5,
            shuffle=True, random_state=0, verbose=2)
    else:
        S_train, S_all = stacking(models, X, y, X_all, sample_weight=sample_weight,
            regression=True, metric=smse, n_folds=5,
            shuffle=True, random_state=0, verbose=2)
        kwargs.pop("sample_weight")

    # Initialize 2-nd level model
    meta_model = meta_learner(**kwargs)

    # Fit 2-nd level model
    meta_model = meta_model.fit(S_train, y)
    y_all = meta_model.predict(S_all)
    return y_all


def skstack(X_train: np.ma.MaskedArray,
            y_train: np.ma.MaskedArray,
            models: List[Tuple[str, TagsMixin]],
            meta_learner: np.ndarray,
            **kwargs: Union[Any, None]) -> Tuple[StackingTransformer, RegressorMixin]:
    """
    Trains the super-learner on the full training data set.
    using vecstack Scikit-learn API (i.e., the StackingTransformer() class) and
    makes predictions using the predictions from the base-learners as the features.

    Parameters
    ----------
    X_train: 2D array
        The training set feature values of covariates (columns) for the samples (rows)
        obtained from the intersection of shapefile with covariate TIFs.
    y_train: 1D array
        The training set target values from the shapefile.
    models: List[Tuple[str, TagsMixin]]
         The list of base-learners (actually pairs, i.e.,  (name, base-learner))
    meta_learner: RegressorMixin
         The super-learner class, one of the 4 Regressors (LinReg, XGBReg, GBM, RF)
    **kwargs: any (optional)
        The optional keyword parameters for meta-learner initialisation

    Returns
    ----------
    stack: RegressorMixin
        The trained super-learner, ready to 
    meta_model: RegressorMixin
        The trained super-learner, ready to make predictions on the tranformed features
    """

    sample_weight = kwargs.get("sample_weight")
    # Initialize StackingTransformer
    stack = StackingTransformer(models, regression=True, verbose=2)
    # Fit
    try:
        stack = stack.fit(X_train, y_train, sample_weight)
    except ValueError as _e:
        _logger.error(f"{_e}")
        stack = stack.fit(X_train, y_train)
    # Get your stacked features
    S_train = stack.transform(X_train)
    # Initialize 2nd level model
    meta_model = meta_learner(**kwargs)
    # Train the 2nd level model
    meta_model = meta_model.fit(S_train, y_train)
    return stack, meta_model


def m_stack(X_train: np.ma.MaskedArray,
            y_train: np.ma.MaskedArray,
            models: List[TagsMixin],
            meta_learner: RegressorMixin,
            **kwargs: Union[Any, None]) -> SuperLearner:
    """
    Trains the super-learner on the full training data set.
    using MLEns SuperLearner()
    makes predictions using the predictions from the base-learners as the features.

    Parameters
    ----------
    X_train: 2D array
        The training set feature values of covariates (columns) for the samples (rows)
        obtained from the intersection of shapefile with covariate TIFs.
    y_train: 1D array
        The training set target values from the shapefile.
    models: List[TagsMixin]
         The list of base-learners
    meta_learner: RegressorMixin
         The super-learner class, one of the 4 Regressors (LinReg, XGBReg, GBM, RF)
    **kwargs: any
        The optional keyword parameters for meta-learner initialisation

    Returns
    ----------
    ensemble: SuperLearner
        The trained super-learner model.
    """

    ensemble = SuperLearner(scorer=smse, folds=5, shuffle=True, sample_size=X_train.shape[0])
    ensemble.add(models)
    ensemble.add_meta(meta_learner(**kwargs))
    # Fit 2-nd level model
    ensemble.fit(X_train, y_train)
    return ensemble
