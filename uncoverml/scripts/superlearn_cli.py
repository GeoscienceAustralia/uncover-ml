#################################################
# Author: Ante Bilic                            #
# Since: 16/02/2022                             #
# Copyright: Geoscience Australia uncover-ML    #
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
from pathlib import Path
import warnings
from typing import List, Tuple, Dict, Union, Any
import joblib

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from revrand.metrics import smse, lins_ccc
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.base import RegressorMixin
from xgboost.sklearn import XGBRegressor
from vecstack import stacking, StackingTransformer
from mlens.ensemble import SuperLearner

from uncoverml.config import Config
import uncoverml.scripts
from uncoverml.validate import regression_validation_scores
from uncoverml.geoio import RasterioImageSource, ImageWriter, get_image_spec
from uncoverml.features import extract_subchunks
from uncoverml.predict import _get_data
from uncoverml.targets import Targets
import uncoverml.mpiops
import uncoverml.learn

from uncoverml.krige import krig_dict
from uncoverml.models import modelmaps, TagsMixin
from uncoverml.optimise.models import transformed_modelmaps

_logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)

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
    ... : Dict[model, config]
        The trained base-learner and its config parameters
    """

    with open(model_file, 'rb') as fin:
        return joblib.load(fin)


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
    except KeyError as _e:
        if msg is None:
            msg = f"Required parameter {_k} not present in the config.\n{_e}"
        _logger.exception(msg)
        raise KeyError from _e


def main(pipeline_file: str, partitions: int) -> None:
    """
    The super-learning driver routine.
    It trains all the base learners on the training data set (interesction of the shapefile
    and covariates from geotifs) and then invokes base_predict() to
    make predictions.

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

    def read_yaml(pipeline_file: str) -> Dict[str, Any]:
        """
        A simple wrapper to be called by the Node 0 and brodcast config_dic

        """
        with open(pipeline_file, 'r', encoding="utf8") as fin:
            try:
                config_dic = yaml.safe_load(fin)
            except UnicodeDecodeError:
                if pipeline_file.endswith('.model'):
                    _logger.error("You're attempting to run uncoverml but have provided the "
                                  "'.model' file instead of the '.yaml' config file. The predict "
                                  "now requires the configuration file and not the model. Please "
                                  "try rerunning the command with the configuration file.")
                else:
                    _logger.error("Can't parse the yaml file. Ensure you've provided the correct "
                                  "file as config file and that the YAML is valid.")
        return config_dic

    # config_dic = uncoverml.mpiops.run_once(read_yaml, pipeline_file)
    config_dic = read_yaml(pipeline_file)
    if 'metalearning' in config_dic:
        meta_alg = config_dic['metalearning']['algorithm']
        meta_learner = meta_map.get(meta_alg)
        try:
            meta_args = config_dic['metalearning']['arguments']
        except KeyError as _e:
            meta_args = {}
        if meta_args is None:
            # empty "arguments:" block
            meta_args = {}
        _logger.info(f"Using meta-learner {meta_map.get(meta_alg)} with\n{meta_args}")
    else:
        meta_learner = LinearRegression
        meta_args = {}
        _logger.info("Using the default meta-learner LinearRegression")

    learn_alg_lst = _grp(config_dic, 'learning', "'learning' block required when metalearning.")
    config_dic.pop("learning")
    # prepare inputs for the base-learners
    yaml_dic = uncoverml.mpiops.run_once(base_yaml, learn_alg_lst, config_dic)
    # synchronise the multi-processing
    # uncoverml.mpiops.comm.barrier()
    # the list of base-learner models (each as an object initialised with class(parameters)):
    base_learner_lst = []
    # the list of trained base-learner models (each as a dict {"model": ... , "config": ...}):
    model_lst = []
    dataset_dic = {}
    # get the training data sets
    for alg, yml in yaml_dic.items():
        a_conf = Config(yml)
        targets_shp, x_shp = uncoverml.scripts.uncoverml._load_data(a_conf, partitions)
        if uncoverml.mpiops.chunk_index == 0:
            dataset_dic.update({alg: (targets_shp, x_shp)})
    dataset_dic = uncoverml.mpiops.comm.bcast(dataset_dic, root=0)
    # train the base-learners
    base_fit(learn_alg_lst, base_learner_lst, model_lst, config_dic, dataset_dic, partitions)
    # proceed to make base model predictions
    df_pred = base_predict(learn_alg_lst, model_lst, config_dic, partitions)
    _logger.info(f"\nBase learner training set predictions:\n{df_pred.head()}")
    _logger.info(f"...\n{df_pred.tail()}")
    # combine the valid base model predictions
    tif_chunk = combine_pred(learn_alg_lst, base_learner_lst, model_lst, df_pred)
    ## To combine chunks from all processes into a single x_tif array:
    # chunk_lst = uncoverml.mpiops.comm.gather(tif_chunk, root=0)
    # if uncoverml.mpiops.chunk_index == 0:
    #     x_tif = np.concatenate(chunk_lst)
    # else:
    #     x_tif = None
    # x_tif = uncoverml.mpiops.comm.bcast(x_tif, root=0)
    ## To continue with separate chunks for each process:
    x_tif = tif_chunk
    # use these to train the first meta-learner and save its predictions
    go_meta(df_pred, x_tif, model_lst[-1], config_dic, meta_learner, **meta_args)
    # train the meta-learner using vecstack and MLEns and make the predictions
    go_vecml(learn_alg_lst,
            base_learner_lst,
            model_lst[-1],
            targets_shp,
            x_shp,
            partitions,
            meta_learner,
            **meta_args)


def base_yaml(learn_alg_lst: List[Dict[str, Any]], config_dic: Dict[str, Any]) -> Dict[str, str]:
    """
    Writes the base-learner yaml files.

    Parameters
    ----------
    learn_alg_lst: List
        the list of base-learner algorithms (each as a dict) with their arguments
    config_dic: dict
        The settings from YAML input file with the definitions of meta-learner,
        base-learners with the associated keyword arguments, covariates, targets,
        validation, prediction etc...

    Return
    ----------
    yaml_dic: Dict[str, str]
        For each base-learner algorithm (key) the associated YAML file (value).
    """

    yaml_dic = {}
    for alg in learn_alg_lst:
        alg_yaml = f"./{alg['algorithm']}.yml"
        alg_outdir = f"./{alg['algorithm']}_out"
        # begin to collect the base-learner parameters in learner_dic:
        learner_dic = {"learning": alg}
        learner_dic.update(config_dic)
        try:
            learner_dic.pop("metalearning")
        except KeyError:
            # if no "metalearning", LinearRegression() used as the super-learner
            pass
        learner_dic["output"]["directory"] = alg_outdir
        learner_dic["output"]["model"] = f"{alg['algorithm']}.model"
        learner_dic["pickling"]["covariates"] = alg_outdir + "/features.pk"
        learner_dic["pickling"]["targets"] = alg_outdir + "/targets.pk"
        # save the base-learner parameters as a YAML file with the same name
        with open(alg_yaml, 'w', encoding="utf8") as yout:
            yaml.dump(learner_dic, yout, default_flow_style=False, sort_keys=False)
        _logger.info(f"\nBase model {alg['algorithm']} yaml written")
        yaml_dic.update({alg['algorithm']: alg_yaml})
    # else:
        # yml_lst = [Path(f"./{_al['algorithm']}.yml") for _al in learn_alg_lst]
        # while True:
            # if all([_yml.exists() and (_yml.stat().st_size > 0) for _yml in yml_lst]):
                # return
            # else:
                # continue
    return yaml_dic


def learn(alg_conf: Config, targets_shp: Targets, x_shp: np.ma.MaskedArray) -> None:
    """
    The function modified from uncoverml/scripts/uncoverml.py so that it
    does not load the same targets_shp and x_shp for each base-learner repeatedly.

    Parameters
    ----------
    alg_conf: Config
        The configurations settings for the present base-learner
    targets_shp: Targets
        the attributes (values, weights, lat_lon, etc) of the dependent/target variable
        from the shapefile
    x_shp: maskedarray
        the values of the covariates/features obtained from the intersection of the
        geotifs with the shapefile
    """

    if alg_conf.cross_validate:
        crossval_results = uncoverml.validate.local_crossval(x_shp, targets_shp, alg_conf)
        uncoverml.mpiops.run_once(uncoverml.geoio.export_crossval, crossval_results, alg_conf)

    _logger.info("Learning full {} model".format(alg_conf.algorithm))
    model = uncoverml.learn.local_learn_model(x_shp, targets_shp, alg_conf)

    uncoverml.mpiops.run_once(uncoverml.geoio.export_model, model, alg_conf)

    # use trained model
    if alg_conf.permutation_importance:
        uncoverml.mpiops.run_once(uncoverml.validate.permutation_importance, model, x_shp,
                           targets_shp, alg_conf)

    # if alg_conf.feature_importance:
    #     # model, x_shp, targets_shp, alg_conf: Config
    #     uncoverml.mpiops.run_once(uncoverml.validate.plot_feature_importance, model, x_shp,
    #                        targets_shp, alg_conf)


def validate(alg_conf: Config, model: TagsMixin, partitions) -> None:
    """
    Adapted from uncoverml.scripts.uncoverml.py.
    Validates a base model with out of sample shapefile.

    Parameters
    ----------
    alg_conf: Config
        The configurations settings for the present base-learner
    model: TagsMixin
        A trained base-learner model
    partitions: int
        The number of data partitions
    """

    alg_conf.pickle_load = False
    alg_conf.target_file = alg_conf.oos_validation_file
    alg_conf.target_property = alg_conf.oos_validation_property
    targets_oos, x_oos = uncoverml.scripts.uncoverml._load_data(alg_conf, partitions)
    uncoverml.validate.oos_validate(targets_oos, x_oos, model, alg_conf)
    _logger.info("Completed OOS validation.")


def base_fit(learn_alg_lst: List[Dict[str, Any]],
            base_learner_lst: List[TagsMixin],
            model_lst: List[Dict[str, Union[TagsMixin, Config, str]]],
            config_dic: Dict[str, Any],
            dataset_dic: Dict[str, Tuple[Targets, np.ma.MaskedArray]],
            partitions: int) -> None:
    """
    Trains the base models.

    Parameters
    ----------
    learn_alg_lst: List
        the list of base-learner algorithms (each as a dict) with their arguments
    base_learner_lst: List
        the list of base-learner models
        (each as a TagsMixnin object initialised with class(parameters)):
    model_lst: List
        the list of trained base-learner models
        each as a dict {"model": ... , "config": ..., "outdir": ..., "mfile": ...)}
    config_dic: dict
        The settings from YAML input file with the definitions of meta-learner,
        base-learners with the associated keyword arguments, covariates, targets,
        validation, prediction etc...
    dataset_dic: Dict[str, Tuple [Targets, maskedarray]]
        for each base-learner algorithm it's target and feature data set
    partitions: int
        The number of data partitions
    """

    for alg in learn_alg_lst:
        _logger.info(f"\nBase model {alg['algorithm']} learning about to begin...")
        alg_yaml = f"./{alg['algorithm']}.yml"
        alg_outdir = f"./{alg['algorithm']}_out"
        # define the base-learner with the YAML file parameters & append it to the list
        base_learner_lst.append(define_model(alg_yaml))
        # train the base-learner
        if Path(alg_yaml).exists() and (Path(alg_yaml).stat().st_size > 0):
            alg_conf = Config(alg_yaml)
            alg_conf.n_subchunks = partitions
            try:
                targets_shp, x_shp = dataset_dic.get(alg['algorithm'])
                learn(alg_conf, targets_shp, x_shp)
            except Exception as _e:
                _logger.error(f"Problem with training {alg['algorithm']}: {_e}")
                raise LearnerError(alg_conf, "Either bad config params or shapefile") from _e

        # load the trained base-model from the *.model file using the Node 0
        alg_model = f"{alg['algorithm']}.model"
        model_conf = uncoverml.mpiops.run_once(load_model, alg_outdir + f"/{alg_model}")
        # add the *.model file location information to its dictionary:
        model_conf.update({"outdir": alg_outdir, "mfile": alg_model})
        # append it to the model list
        model_lst.append(model_conf)

        if hasattr(alg_conf,'oos_validation_file') and hasattr(alg_conf, 'oos_validation_property'):
            validate(alg_conf, model_conf["model"], partitions)


class LearnerError(Exception):
    """Exception raised for errors in the input in a base-learner YAML file.

    Attributes:
        alg_conf -- model input YAML config
        message  -- explanation of the error
    """

    def __init__(self, alg_conf: str, message: str="Config paramater(s) invalid"):
        self.alg_conf = alg_conf
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.alg_conf}: {self.message}'


def base_predict(learn_alg_lst: List[Dict[str, Any]],
                model_lst: List[Dict[str, Union[TagsMixin, Config, str]]],
                config_dic: Dict[str, Any],
                partitions: int) -> pd.DataFrame:
    """
    Makes the predictions from the trained base models

    Parameters
    ----------
    learn_alg_lst: List
        the list of base-learner algorithms (each as a dict) with their arguments
    model_lst: List
        the list of trained base-learner models
        each as a dict {"model": ... , "config": ..., "outdir": ..., "mfile": ...)}
    config_dic: dict
        The settings from YAML input file with the definitions of meta-learner,
        base-learners with the associated keyword arguments, covariates, targets,
        validation, prediction etc...
    partitions: int
        The number of data partitions

    Return
    ----------
    df_pred: DataFrame
        True target values ("y_true") from the shapefile and predictions from the base models
    """

    df_pred = pd.DataFrame()

    for alg, model in zip(learn_alg_lst, model_lst):
        retain = None
        _mb = config_dic.get('mask')
        if _mb:
            mask = _mb.get('file')
            if not Path(mask).exists():
                raise FileNotFoundError("Mask file provided in config does not exist.")
            retain = _grp(_mb, 'retain', "'retain' must be provided if prediction mask applied.")
        else:
            mask = None

        _logger.info(f"Base learner {alg['algorithm']} predictions about to begin...")
        model_file = model["outdir"] + "/" + model["mfile"]
        try:
            uncoverml.scripts.uncoverml.predict.callback(model_file, partitions, mask, retain)
        except TypeError:
            _logger.error(f"Learner {alg['algorithm']} cannot predict")

        if "validation" in config_dic:
            pred = pd.read_csv(model["outdir"] + "/" + f"{alg['algorithm']}_results.csv")
            pred.rename(columns={'y_pred': alg['algorithm']}, inplace=True)
            pred.drop(columns=['y_transformed'], inplace=True)
        elif "oos_validation" in config_dic:
            pred = pd.read_csv(model["outdir"] + "/" + f"{alg['algorithm']}_oos_validation.csv",
                    usecols=['Prediction', 'y_true', 'lon', 'lat'])
            pred.rename(columns={'Prediction': alg['algorithm']}, inplace=True)
        duplicates = pred[['y_true', 'lon', 'lat']].duplicated().sum()
        if duplicates:
            _logger.info(f"Base learner {alg['algorithm']} produces {duplicates} duplicated rows.")
        if df_pred.empty:
            df_pred = pred
        else:
            # due to shuffle=True the rows must be matched on y_true, lat & lon
            df_pred = df_pred.merge(pred, on=['y_true', 'lat', 'lon'])

    df_pred.drop(columns=['lat', 'lon'], inplace=True)
    return df_pred


def combine_pred(learn_alg_lst: List[Dict[str, Any]],
                base_learner_lst: List[TagsMixin],
                model_lst: List[Dict[str, Union[TagsMixin, Config, str]]],
                df_pred: pd.DataFrame) -> np.ma.MaskedArray:
    """
    Combines the base-model learner predictions into a set of features for the meta-learner.
    It drops the base-learners/models with invalid (NaN, Inf...) target predictions.

    Parameters
    ----------
    learn_alg_lst: List
        the list of base-learner algorithms (each as a dict) with their arguments
    base_learner_lst: List
        the list of base-learner models
        (each as a TagsMixnin object initialised with class(parameters)):
    model_lst: List
        the list of trained base-learner models
        each as a dict {"model": ... , "config": ..., "outdir": ..., "mfile": ...)}
    df_pred: DataFrame
        True target values ("y_true") from the shapefile and predictions from the base models

    Return
    ----------
    x_tif: 2D MaskedArray
        The predictor features for the super-learner
    """

    base_alg_lst = [alg["algorithm"] for alg in learn_alg_lst]
    # combine the base model predictions (in the form of geotifs) into a 2D array
    x_tif = get_metafeats(base_alg_lst)
    # drop the base-learner(s) that produce invalid target values like NaN or Inf
    column_nan = np.logical_or(np.isnan(x_tif).any(axis=0), np.isinf(x_tif).any(axis=0))
    if column_nan.any():
        _logger.info(f"Inspect learner rasters for NaNs: {*column_nan,}")
        colnan_ix = np.argwhere(column_nan)
        _logger.info(f"Problem column(s) with NaNs: {*colnan_ix,}")
        x_tif = np.delete(x_tif, colnan_ix[0], axis=1)
        for _i in np.argwhere(column_nan)[0][::-1]:
            learn_alg_lst.pop(_i)
            model_lst.pop(_i)
            base_learner_lst.pop(_i)
        df_pred.drop(df_pred.columns[colnan_ix[0]], axis=1, inplace=True)

    return x_tif


def go_meta(df_pred: pd.DataFrame,
            x_tif: np.ma.MaskedArray,
            model_conf: List[Dict[str, Union[TagsMixin, Config, str]]],
            config_dic: Dict[str, Any],
            meta_learner: RegressorMixin,
            **meta_args: Union[Any, None],
            ) -> None:
    """
    Combines the base model predictions from the training set into the training features
    for the meta-learner, carries out the cross validation on those, and uses them
    to train the meta-learner. The final predictions are saved as 'Meta_predict.tif'
    geotif file.

    Parameters
    ----------
    df_pred: DataFrame
        True target values ("y_true") from the shapefile and predictions from the base models
    x_tif: 2D MaskedArray
        The predictor features for the super-learner
    model_conf: Dict
        any trained base-learner and its config
    config_dic: Dict
        The settings from the input file
    meta_learner: RegressorMixin
        The super-learner type
    **meta_args: any
        The optional keyword parameters for meta-learner initialisation
    """

    y_mtrain = df_pred['y_true'].values
    x_mtrain = df_pred.drop(columns='y_true').values
    if 'validation' in config_dic:
        try:
            cvfs = [int(_lst[0]['k-fold']['folds']) for _lst in config_dic['validation']
                        if "k_fold" in _lst[0]][0]
        except (KeyError, TypeError, IndexError) as _e:
            _logger.exception(f"CV fold unreadable {_e}")
            cvfs = 5
    else:
        cvfs = 5
    _logger.info(f"meta-learner CV with {cvfs} folds")

    # meta-learner CV
    uncoverml.mpiops.run_once(meta_cv, x_mtrain, y_mtrain, meta_learner, n_splits=cvfs, **meta_args)
    # meta-learner training
    meta_model = uncoverml.mpiops.run_once(meta_fit, x_mtrain, y_mtrain, meta_learner, **meta_args)
    # meta-learner predict
    y_meta_tif = meta_predict(meta_model, x_tif)
    target_tif = meta_tif(model_conf, "Meta", 1)
    target_tif.write(y_meta_tif.view(np.ma.masked_array).reshape(-1, 1), 0)
    target_tif.close()


def go_vecml(learn_alg_lst: List[Dict[str, Any]],
            base_learner_lst: List[TagsMixin],
            model_conf: List[Dict[str, Union[TagsMixin, Config, str]]],
            targets_shp: Targets,
            x_shp: np.ma.MaskedArray,
            partitions: int,
            meta_learner: RegressorMixin,
            **meta_args: Union[Any, None],
            ) -> None:
    """
    Makes the predictions from the trained base models using vecstack and MLEns
    and saves those as geotifs.

    Parameters
    ----------
    learn_alg_lst: List
        the list of base-learner algorithms (each as a dict) with their arguments
    base_learner_lst: List
        the list of base-learner models
        (each as a TagsMixnin object initialised with class(parameters)):
    model_conf: Dict
        Any trained base-model and its config
    targets_shp: Targets
        the attributes (values, weights, lat_lon, etc) of the dependent/target variable
        from the shapefile
    x_shp: maskedarray
        the values of the covariates/features obtained from the intersection of the
        geotifs with the shapefile
    partitions: int
        The number of data partitions
    meta_learner: RegressorMixin
        The super-learner type
    **meta_args: any
        The optional keyword parameters for meta-learner initialisation
    """

    y_shp = targets_shp.observations
    weights = targets_shp.weights
    # initialize ImageWriter objects
    starget_tif = meta_tif(model_conf, "VSuper", partitions)
    mtarget_tif = meta_tif(model_conf, "MSuper", partitions)
    tup = [(alg['algorithm'], _bl) for alg, _bl in zip(learn_alg_lst, base_learner_lst)]
    # vecstack & MLEns learn, on Node 0 and broadcast
    stack, meta_model = uncoverml.mpiops.run_once(skstack,
                                                x_shp,
                                                y_shp,
                                                tup,
                                                meta_learner,
                                                sample_weight=weights,
                                                **meta_args)
    mle_stack = uncoverml.mpiops.run_once(m_stack,
                                        x_shp,
                                        y_shp,
                                        base_learner_lst,
                                        meta_learner,
                                        **meta_args)
    for _i in range(partitions):
        # NOTE: each node will produce different x_ip feature set
        x_ip, _ = _get_data(_i, model_conf["config"])
        # vecstack transform raw features
        s_ip = stack.transform(x_ip)
        # vecstack predict across the partition _i
        y_vstack_tif = meta_model.predict(s_ip)
        # write predictions for the partition _i
        starget_tif.write(y_vstack_tif.view(np.ma.masked_array).reshape(-1, 1), _i)
        # MLEns predict across the partition _i
        y_mstack_tif = mle_stack.predict(x_ip)
        # write predictions for the partition _i
        mtarget_tif.write(y_mstack_tif.view(np.ma.masked_array).reshape(-1, 1), _i)
    starget_tif.close()
    mtarget_tif.close()


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
    x_all: 2D MaskedArray
        The final predictor features for the super-learner
    """

    arr_lst = []
    for tif in tif_lst:
        img = RasterioImageSource(tif)
        marr = extract_subchunks(img, 0, 1, 0)
        arr_lst.append(marr[:, 0, 0, 0])

    x_all = np.column_stack(arr_lst)
    return x_all


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
    x_tif: 2D MaskedArray
        The predictor features for the super-learner
    """

    arr_lst = []
    for alg in base_alg_lst:
        alg_outdir = f"./{alg}_out"
        tif = alg_outdir + f"/{alg}_prediction.tif"
        img = RasterioImageSource(tif)
        marr = extract_subchunks(img, 0, 1, 0)
        arr_lst.append(marr[:, 0, 0, 0])

    x_tif = np.column_stack(arr_lst)
    return x_tif


def meta_cv(x_mtrain: np.ma.MaskedArray,
            y_mtrain: np.ma.MaskedArray,
            meta_learner: RegressorMixin,
            n_splits: int=5,
            **kwargs: Union[Any, None]) -> None:
    """
    Runs the cross validation for the super-learner.

    Parameters
    ----------
    x_mtrain: 2D array
        The training set feature values are the predictions of the base-learners
        over the samples obtained from the intersection of shapefile with covariate TIFs.
    y_mtrain: 1D array
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

    for train_ix, test_ix in kfold.split(x_mtrain):
        train_x, test_x = x_mtrain[train_ix], x_mtrain[test_ix]
        train_y, test_y = y_mtrain[train_ix], y_mtrain[test_ix]
        _sm = MetaLearn(meta_learner, **kwargs)
        _sm.fit(train_x, train_y)
        y_oof = _sm.predict(test_x)
        y_pred.extend(y_oof)
        y_oof = y_oof.reshape(-1, 1)
        fold_scores.append(regression_validation_scores(test_y, y_oof, np.ones_like(y_oof), _sm))
    y_pred = np.array(y_pred)
    _logger.info(f"\nSUPER CV r2 = {r2_score(y_mtrain, y_pred)}")
    _logger.info(f"\nSUPER CV expvar = {explained_variance_score(y_mtrain, y_pred)}")
    _logger.info(f"\nSUPER CV smse = {smse(y_mtrain, y_pred)}")
    _logger.info(f"\nSUPER CV lins_ccc = {lins_ccc(y_mtrain, y_pred)}")
    _logger.info(f"\nSUPER CV mse = {mean_squared_error(y_mtrain, y_pred)}")
    scores_df = pd.DataFrame(fold_scores)
    _logger.info(f"\nSUPER CV scores:\n{scores_df}")
    _logger.info("\nSUPER CV score averages:")
    for _c in scores_df.columns:
        _logger.info(f"{_c} = {scores_df[_c].mean()}")
    plt.figure()
    plt.scatter(y_mtrain, y_pred)
    plt.xlabel("True target")
    plt.ylabel("Meta-learner predicted")
    plt.savefig("Meta_cv_predicted.png")
    plt.close()


def meta_fit(x_mtrain: np.ma.MaskedArray,
            y_mtrain: np.ma.MaskedArray,
            meta_learner: RegressorMixin,
            **kwargs: Union[Any, None]) -> RegressorMixin:
    """
    Trains the super-learner on the full training data set.

    Parameters
    ----------
    x_mtrain: 2D array
        The training set feature values are the predictions of the base-learners
        over the samples obtained from the intersection of shapefile with covariate TIFs.
    y_mtrain: 1D array
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
    meta_model.fit(x_mtrain, y_mtrain)
    if isinstance(meta_model, LinearRegression):
        _logger.info(f"SUPER coeffs:\n{meta_model.model.coef_}")
        _logger.info(f"SUPER intercept:\n{meta_model.model.intercept_}")
    y_pred = meta_model.model.predict(x_mtrain[:, :])
    _logger.info(f"\nSUPER smse: {smse(y_mtrain, y_pred)}")
    _logger.info(f"\nSUPER r2: {r2_score(y_mtrain, y_pred)}")
    plt.figure()
    plt.scatter(y_mtrain, y_pred)
    plt.xlabel("True target")
    plt.ylabel("Meta-learner predicted")
    plt.savefig("Meta_train_predicted.png")
    plt.close()
    return meta_model


def meta_predict(meta_model: RegressorMixin, x_tif: np.ma.MaskedArray) -> np.ndarray:
    """
    Makes predictions using the trained super-learner and the predictions from the
    base-learners as the features.

    Parameters
    ----------
    model: RegressorMixin
        The trained super-learner model.
    x_tif: 2D MaskedArray
        The predictor features for the super-learner with columns consisting of
        the predictions of base-learners and rows covering the area represented
        by the geotifs.

    Returns
    ----------
    y_tif: 1D array
        The trained super-learner model final predictions.
    """

    y_tif = meta_model.predict(x_tif)
    _logger.debug(f"Feature-target shapes: {x_tif.shape}, {y_tif.shape}")
    return y_tif


def meta_tif(model_config: Dict[str, Union[TagsMixin, Config, str]],
            tif_name: str="Meta",
            partitions: int=1) -> ImageWriter:
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

    Returns
    ----------
    img_out: ImageWriter
        The image object to close when all partitions chunks are written
    """

    shape, bbox, crs = get_image_spec(model_config["model"], model_config["config"])
    img_out = ImageWriter(shape, bbox, crs, tif_name, partitions, "./", band_tags=["Prediction"])
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


    def fit(self, _x: np.ndarray, _y: np.ndarray) -> None:
        """
        Trains the super-learner on the full training data set.

        Parameters
        ----------
        _x: 2D array
            The training set feature values are the predictions of the base-learners
            over the samples obtained from the intersection of shapefile with covariate TIFs.
        _y: 1D array
            The training set target values from the shapefile.
        """

        self.model.fit(_x, _y)


    def predict(self, x_tif: np.ma.MaskedArray) -> np.ndarray:
        """
        Makes predictions using the trained super-learner and the predictions from the
        base-learners as the features.

        Parameters
        ----------
        x_tif: 2D MaskedArray
            The predictor features for the super-learner

        Returns
        ----------
        : 1D array
            The training set target values from the shapefile.
        """

        return self.model.predict(x_tif)


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

            super().__init__(**kwargs)


        def fit(self, _x: np.ndarray, _y: np.ndarray) -> None:
            """
            Trains the super-learner on the full training data set.

            Parameters
            ----------
            _x: 2D array
                The training set feature values are the predictions of the base-learners
                over the samples obtained from the intersection of shapefile with covariate TIFs.
            _y: 1D array
                The training set target values from the shapefile.
            """

            super().fit(_x, _y)


        def predict(self, x_tif: np.ma.MaskedArray) -> np.ndarray:
            """
            Makes predictions using the trained super-learner and the predictions from the
            base-learners as the features.

            Parameters
            ----------
            x_tif: 2D MaskedArray
                The predictor features for the super-learner

            Returns
            ----------
            ...: 1D array
                The training set target values from the shapefile.
            """

            return super().predict(x_tif)


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


def v_stack(x_mtrain: np.ndarray,
            y_mtrain: np.ndarray,
            x_all: np.ndarray,
            models: List[TagsMixin],
            meta_learner: np.ndarray,
            **kwargs: Union[Any, None]) -> np.ndarray:
    """
    Trains the super-learner on the full training data set.
    using vecstack Functional API (i.e., the stacking() function) and
    makes predictions using the predictions from the base-learners as the features.

    Parameters
    ----------
    x_mtrain: 2D array
        The training set feature values of covariates (columns) for the samples (rows)
        obtained from the intersection of shapefile with covariate TIFs.
    y_mtrain: 1D array
        The training set target values from the shapefile.
    x_all: 2D array
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
        s_train, s_all = stacking(models, x_mtrain, y_mtrain, x_all,
            regression=True, metric=smse, n_folds=5,
            shuffle=True, random_state=0, verbose=2)
    else:
        s_train, s_all = stacking(models, x_mtrain, y_mtrain, x_all, sample_weight=sample_weight,
            regression=True, metric=smse, n_folds=5,
            shuffle=True, random_state=0, verbose=2)
        kwargs.pop("sample_weight")

    # Initialize 2-nd level model
    meta_model = meta_learner(**kwargs)

    # Fit 2-nd level model
    meta_model = meta_model.fit(s_train, y_mtrain)
    y_all = meta_model.predict(s_all)
    return y_all


def skstack(x_train: np.ma.MaskedArray,
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
    x_train: 2D array
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
    stack: StcakingTransformer
        The trained transformer, needed to transform the raw features
    meta_model: RegressorMixin
        The trained super-learner, ready to make predictions on the tranformed features
    """

    sample_weight = kwargs.get("sample_weight")
    # Initialize StackingTransformer
    stack = StackingTransformer(models,
                                regression=True,
                                variant='A',
                                metric=smse,
                                n_folds=5,
                                shuffle=True,
                                random_state=42,
                                verbose=2)
    # Fit
    try:
        stack = stack.fit(x_train, y_train, sample_weight)
    except ValueError as _e:
        _logger.error(f"Sample weight problem: {_e}")
        stack = stack.fit(x_train, y_train)
    # Get your stacked features
    s_train = stack.transform(x_train)
    # Initialize 2nd level model
    try:
        meta_model = meta_learner(**kwargs)
    except TypeError as _e:
        _logger.error(f"Sample weight problem: {_e}")
        kwargs.pop("sample_weight")
        meta_model = meta_learner(**kwargs)
    # Train the 2nd level model
    meta_model = meta_model.fit(s_train, y_train)
    return stack, meta_model


def m_stack(x_train: np.ma.MaskedArray,
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
    x_train: 2D array
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

    ensemble = SuperLearner(scorer=smse, folds=5, shuffle=True, sample_size=x_train.shape[0])
    ensemble.add(models)
    ensemble.add_meta(meta_learner(**kwargs))
    # Fit 2-nd level model
    ensemble.fit(x_train, y_train)
    return ensemble
