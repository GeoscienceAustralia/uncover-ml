"""
Handles parsing of the configuration file.
"""
from typing import Optional, List
import logging
from os import path
from os import makedirs
import os
import glob
import csv
import re

import yaml

from uncoverml import transforms

_logger = logging.getLogger(__name__)

"""
The strings associated with each imputation option.
"""
_imputers = {'mean': transforms.MeanImputer,
             'gaus': transforms.GaussImputer,
             'nn': transforms.NearestNeighboursImputer}

"""
These transforms operate on each image before concatenation.
"""
_image_transforms = {'onehot': transforms.OneHotTransform,
                     'randomhot': transforms.RandomHotTransform}

"""
Post-concatenation transforms: operate on whole data vector.
"""
_global_transforms = {'centre': transforms.CentreTransform,
                      'standardise': transforms.StandardiseTransform,
                      'log': transforms.LogTransform,
                      'sqrt': transforms.SqrtTransform,
                      'whiten': transforms.WhitenTransform}

    

def _parse_transform_set(transform_dict: dict, imputer_string: str, n_images: int=None) -> tuple:
    """
    Parse a dictionary read from yaml into a TransformSet object.

    Parameters
    ----------
    transform_dict
        The dictionary as read from the yaml config file containing
        config key-value pairs.
    imputer_string
        The name of the imputer.
    n_images
        The number of images being read in. Required because we need to
        create a new image transform for each image.
        
    Returns
    -------
    image_transforms
        A list of image Transform objects.
    imputer
        An Imputer object.
    global_transforms
        A list of global Transform objects.
    """
    image_transforms = []
    global_transforms = []
    if imputer_string is not None and imputer_string in _imputers:
        imputer = _imputers[imputer_string]()
    else:
        imputer = None
    if transform_dict is not None:
        for t in transform_dict:
            if type(t) is str:
                t = {t: {}}
            key, params = list(t.items())[0]
            if key in _image_transforms:
                image_transforms.append([_image_transforms[key](**params) 
                                        for k in range(n_images)])
            elif key in _global_transforms:
                global_transforms.append(_global_transforms[key](**params))
    return image_transforms, imputer, global_transforms


class FeatureSetConfig(object):
    """
    Config class representing a 'feature set' in the config file.

    Parameters
    ----------
    config_dict
        The section of the yaml file for a feature set.

    Attributes
    ----------
    name : str
        Name of the feature set.
    type : str
        Data type of the feature set ('categorical' or 'ordinal').
    files : list of str
        Absolute paths to .tif files of the feature set.
    transform_set : :class:`~uncoverml.transforms.transformset.ImageTransformSet`
        Transforms specified for the feautre set.
    """
    def __init__(self, config_dict: dict):
        d = config_dict
        self.name = d['name']
        if d['type'] not in ('ordinal', 'categorical'):
            _logger.warning("Feature set type must be ordinal or categorical. "
                            "Unknwon option: '%s'. Type has been set to 'ordinal'.",
                            d['type'])
            self.type = 'ordinal'
        else:
            self.type = d['type']
        is_categorical = d['type'] == 'categorical'

        # get list of all the files
        files = []
        for source in d['files']:
            key = next(iter(source.keys()))
            if key == 'path':
                files.append(path.abspath(source[key]))
            elif key == 'directory':
                glob_string = path.join(path.abspath(source[key]), "*.tif")
                f_list = glob.glob(glob_string)
                files.extend(f_list)
            elif key == 'list':
                csvfile = path.abspath(source[key])
                with open(csvfile, 'r') as f:
                    reader = csv.reader(f)
                    tifs = list(reader)
                    tifs = [f[0].strip() for f in tifs
                            if (len(f) > 0 and f[0].strip() and
                                f[0].strip()[0] != '#')]
                for f in tifs:
                    files.append(path.abspath(f))

        self.files = sorted(files, key=str.lower)
        n_files = len(self.files)

        trans_i, im, trans_g = _parse_transform_set(d['transforms'],
                                                    d['imputation'],
                                                    n_files)
        self.transform_set = transforms.ImageTransformSet(trans_i, im, trans_g, is_categorical)
                                                          


class Config(object):
    """
    Class representing the global configuration of the uncoverml
    scripts.

    This class is *mostly* read-only, but it does also contain the
    Transform objects which have state.

    .. todo::

        Factor out stateful Transform objects.

    .. todo::

        The mechanism for selecting outbands could be a lot better
        and explanation of the outbands available (or where to find
        that infomration for each model) would be a good addition.

    .. todo::
        
        rawcovariates and train_data_pk are initialised as bools
        but treated as strings (file paths) later in the code

    Parameters
    ----------
    yaml_file
        The path to the yaml config file.

    Attributes
    ----------
    name : str
        Name fo the config file.
    patchsize : int
        Half-width of the patches that feature data will be chunked
        into. Height/width of each patch is equal to patchsize * 2 + 1.

        .. todo::
            
            Not implemneted, defaults to 1.

    algorithm : str
        The ML model to train.
    cubist : bool
        True if the selected model is 'cubist'.
    multicubist : bool
        True if the selected model is 'multicubist'.
    multirandomforest : bool
        True if the selected model is 'multirandomforest'.
    krige : bool
        True if the selected model is 'krige'.
    algorithm_args : dict
        Arguments for the selected model. See the parameters for models
        in the :mod:`~uncoverml.models` module.
    quantiles : float
        Prediction quantile/interval for predicted values.
    geotif_options : dict
        Optional creation options passed to the geotiff output driver.
        See https://gdal.org/drivers/raster/gtiff.html#creation-options
        for a list of creation options.
    outbands : int
        The outbands to write in the prediction output file. Used as
        the 'stop' for a slice taken from list of prediction tags,
        i.e. [0: outbands]. If the resulting slice is greater than
        the number of tags available, then all tags will be selected.
        If no value is provied, then all tags will be selected.
    thumbnails : int
        Subsampling factor for thumbnails of output images. Default
        is 10.
    pickle : bool
        Whether or not a feature set of type 'pickle' is present
        in the config.
    raw_covariates_dir : str, optional
        Path to a directory for saving intersected features and targets
        as CSV before processing is performed (hence 'raw'). Will save
        two files: the intersected values and a mask of intersection
        locations. If not provided will be None.
    pk_covariates : str
        Path to pickle file containing intersection of targets and
        covariates. If :attr:`~pk_load` is True, then this file
        will be loaded and used as covariates for learning. If
        :attr:`~pk_load` is False, and this file does not exist,
        then covariates will be dumped to this file after they have
        been created from the feature sets specified in the 
        configuration.
    pk_targets : str
        Path to pickle file containing targets. If :attr:`~pk_load` is
        True, then this file will be loaded and used as targets for
        learning. If :attr:`~pk_load` is False, and this file does not
        exist, then targets will be dumped to this file after they have
        been created from the target data specified in the 
        configuration.
    pk_load : bool
        True if :attr:`~pickle` is True and existent files are provided
        for :attr:`~pickled_covariates` and :attr:`~pickled_targets`.
        If True, then covariates and targets will be loaded from the
        pickle files. If False, the created targets and covariates
        will be dumped if respective file paths are provided. Will
        also be set to False if :attr:`~rank_features` is True as
        feature ranking is not compatible with pickled data. If using
        Cubist or Multicubist algorithms, it will also require 
        :attr:`~pk_featurevec` to be present and an existing file
        in order to be True.
    pk_featurevec : str
        Path to pickle file containing feature vector. Must be provided
        if algorithm is 'cubist' or 'multicubist' and loading from
        pickle files. 
    feature_sets : list of :obj:`~uncoverml.config.FeatureSetConfig`
        A list of feature sets; one for each non-pickle type feature
        set specified in the config.
    final_transform : :obj:`~uncoverml.transforms.transformset.TransformSet`
        A TransformSet containing an imputer and transforms that is
        applied to all features following image transforms, imputation,
        global transforms and concatenation.
    target_file : str
        Path to a Shapefile containing target values and X, Y 
        coordinates.
    target_propery : str
        Name of the property in the target file to use for training.
    resample : list of dict
        Resampling arguments for target data. 

        .. todo::

            Not yet implemented.
    
    mask : str, optional
        Path to a geotiff file for masking the output prediction 
        map. Only values that have been masked will be predicted.
    retain : int, optional
        Value in the above mask that indicates cell should be retained
        and predicted. Must be provided if a mask is provided.
    lon_lat : dict, optional
        Dictionary containing paths to longitude and latitude grids
        used in kriging.
    rank_features : bool
        True if 'feature_ranking' is present in 'validation' block
        of the config. Turns on feature ranking. Default is False.
    permutation_importance : bool
        True if 'permutation_importance' is present in 'validation'
        block of the config. Turns on permutation importance.  
        Default is False.
    parallel_validate : bool
        True if 'parallel' is present in 'k-fold' block of
        config. Turns on parallel k-fold cross validation. Default
        is False.
    cross_validate : bool
        True if 'k-fold' is present in 'validation' block of config.
        Turns on k-fold cross validation.
    folds : int
        The number of folds to split dataset into for cross validation.
    crossval_seed : int
        Seed for random sorting of folds for cross validation.
    output_dir : str
        Path to directory where prediciton map and other outputs
        will be written.
    model_file : str
        Path to the file where model will be saved after
        learning/clustering and loaded from when predicting.
    scores_file : str
        Path to the JSON file where cross validation scores will be
        saved.
    plot_covaraites_dir : str
        Path to directory where plotted covariates will be stored.
    optimisation : dict
        Dictionary of optimisation arguments.
    clustering : bool
        True if 'clustering' present in config file. 

        .. note:: 

            Only seems to be used in :mod:`~uncoverml.predict`, 
            otherwise clustering is a specific CLI process.

    cluster_analysis : bool
        True if 'cluster_analysis' in the 'clustering' block of the
        config file. Turns on cluster analysis.

        .. todo::
        
            Need a better explanation of what clustering analysis does.

    clustering_algorithm : str
        Name of the clustering algorithm to use.

        .. note::
            
            The only available algorith is kmeans. This is used to set
            :attr:`~algorithm` when performing clustering due to the
            polymorphism of model classes.

        .. todo:: 

            If kmeans is the only option, why not just set 
            :attr:`~algorithm` directly.

    n_classes : int
        The number of cluster centres to be used in clustering. If
        running semisupervised learning and this is set to less than
        the number of labelled classes in the training data, then the
        the number of laballed classes in the training data will be
        used as the value instead.
    oversample_factor : float
        Controls how many candidates are found for cluster 
        initialisation when running kmeans clustering. See
        :func:`~uncoverml.cluster.weighted_starting_candidates`.
    semisupervised : bool
        True if a training data file is provided in clustering 
        arguments. Turns on semisupervised clustering.
    class_file : str
        Path to a Shapefile containing training data for clustering.
    class_property : str
        Name of the property to train clustering against.
    """
    def __init__(self, yaml_file: str, cluster=False):

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

        Config._configure_pyyaml()
        with open(yaml_file, 'r') as f:
            s = yaml.load(f, Loader=Config.yaml_loader)
        self.name = path.basename(yaml_file).rsplit(".", 1)[0]

        # LEARNING BLOCK
        if not cluster:
            learn_block = _grp(s, 'learning')
            self.clustering = False
            self.cluster_analysis = False
            self.algorithm = _grp(learn_block, 'algorithm',
                                  "'algorithm' must be provided as part of 'learning' block.")
            self.cubist = self.algorithm == 'cubist'
            self.multicubist = self.algorithm == 'multicubist'
            self.multirandomforest = self.algorithm == 'multirandomforest'
            self.krige = self.algorithm == 'krige'
            self.algorithm_args = _grp(learn_block, 'arguments',
                                       "'arguments' must be provided for learning algorithm.")
        # CLUSTERING BLOCK
        else:
            cb = _grp(s, 'clustering', "'clustering' block must be provided when clustering.")
            self.clustering = True
            self.n_classes = _grp(cb, 'n_classes', "'n_classes' must be provided when clustering.")
            self.oversample_factor = _grp(cb, 'oversample_factor',
                                          "'oversample_factor' must be provided when clustering.")
            self.cluster_analysis = cb.get('cluster_analysis', False)
            self.class_file = cb.get('file')
            if self.class_file:
                self.class_property = _grp(cb, 'property', "'property' must be provided when "
                                           "providing a file for semisupervised clustering.")
            self.semi_supervised = self.class_file is not None
        
        # PICKLING BLOCK
        pk_block = s.get('pickling')
        if pk_block:
            self.pk_covariates = pk_block.get('covariates')
            self.pk_targets = pk_block.get('targets')

            # Load from pickle files if covariates and targets exist.
            self.pk_load = self.pk_covariates and os.path.exists(self.pk_covariates) \
                           and self.pk_targets and os.path.exists(self.pk_targets)
            
            if self.cubist or self.multicubist:
                self.pk_featurevec = pk_block.get('featurevec')
                # If running multicubist, we also need featurevec to load from pickle files.
                self.pk_load = self.pk_load \
                               and self.pk_featurevec and os.path.exists(self.pk_featurevec)
        else:
            self.pk_load = False

        # FEATURES BLOCK
        if not self.pk_load:
            features = _grp(s, 'features', "'features' block must be provided when not loading "
                            "from pickled data.")
            self.feature_sets = [FeatureSetConfig(f) for f in features]

        # Not yet implemented.
        if 'patchsize' in s:
            _logger.info("Patchsize currently fixed at 0 -- ignoring")
        self.patchsize = 0

        
        # TARGET BLOCK
        if not self.pk_load:
            tb = _grp(s, 'targets', "'targets' block my be provided when not loading from "
                      "pickled data.")
            self.target_file = _grp(tb, 'file', "'file' needs to be provided when specifying "
                                    "targets.")
            self.target_property = _grp(tb, 'property', "'property needs to be provided when "
                                        "specifying targets.")
            self.resample = tb.get('resample')

        # FINAL TRANSFORM BLOCK
        ftb = s.get('final_transform')
        if ftb is not None:
            _, im, trans_g = _parse_transform_set(ftb.get('transforms'), ftb.get('imputation'))
            self.final_transform = transforms.TransformSet(im, trans_g)
        else:
            self.final_transform = None
                
        # VALIDATION BLOCK
        vb = s.get('validation')
        if vb:
            self.rank_features = vb.get('feature_rank', False)
            if self.pk_load and self.rank_features:
                _logger.warning("Feature ranking cannot be performed when loading covariates and "
                                "targets from pickled data.")
                self.rank_features = False
            self.permutation_importance = vb.get('permutation_importance', False)
            kfb = vb.get('k-fold')
            if kfb:
                self.cross_validate = True
                self.folds = _grp(kfb, 'folds', "'folds' (number of folds) must be specified "
                                  "if k-fold cross validation is being used.")
                self.crossval_seed = _grp(kfb, 'random_seed', "'random_seed' must be specified "
                                          "if k-fold cross validation is being used.")
                self.parallel_validate = kfb.get('parallel', False)
        else:
            self.cross_validate = False
            self.rank_features = False
            self.permutation_importance = False
            self.parallel_validate = False

        # OPTIMISATION BLOCK
        # Note: optimisation options get parsed in scripts/gridsearch.py
        self.optimisation = s.get('optimisation')

        # PREDICT BLOCK
        pb = _grp(s, 'prediction', "'prediction' block must be provided.")
        self.geotif_options = pb.get('geotif', {})
        self.quantiles = _grp(pb, 'quantiles', "'quantiles' must be provided as part of "
                              "prediction block.")
        self.outbands = _grp(pb, 'outbands', "'outbands' must be provided as part of prediction "
                             "block.")
        self.thumbnails = pb.get('thumbnails', 10)
        mb = s.get('mask')
        if mb:
            self.mask = mb.get('file') 
            self.mask = None if not os.path.exists(self.mask) else self.mask
            if self.mask:
                self.retain = _grp(mb, 'retain', "'retain' must be provided if providing a "
                                   "prediction mask.")
        else:
            self.mask = None
        
        if self.krige:
            # Todo: don't know if lon/lat is compulsory or not for kriging
            self.lon_lat = s.get('lon_lat')
        else:
            self.lon_lat = None

        ob = _grp(s, 'output', "'output' block is required.")
        self.output_dir = _grp(ob, 'directory', "'directory' for output is required.")
        self.model_file = ob.get('model', os.path.join(
            self.output_dir, self.name + '_' + self.algorithm + '.model'))
        if self.cross_validate:
            self.scores_file = ob.get('scores', os.path.join(
                self.output_dir, self.name + '_' + 'scores.json'))
        self.raw_covariates_dir = ob.get('raw_covariates')
        self.plot_covariates_dir = ob.get('plot_covariates')

        paths = [self.output_dir, os.path.split(self.model_file)[1], 
                 os.path.split(self.scores_file)[1], self.raw_covariates_dir,
                 self.plot_covariates_dir]
        for p in paths:
            if p:
                makedirs(p, exist_ok=True)

    yaml_loader = yaml.SafeLoader
    """The PyYaml loader to use."""

    @staticmethod
    def _configure_pyyaml():
        # Configure PyYaml to implicitly resolve environment variables of form '$ENV_VAR'.
        env_var_pattern = re.compile(r'\$([A-Z_]*)')
        yaml.add_implicit_resolver('!envvar', env_var_pattern, Loader=Config.yaml_loader)

        def _env_var_constructor(loader, node):
            """
            PyYaml constructor for resolving env vars.
            """
            value = loader.construct_scalar(node)
            env_vars = env_var_pattern.findall(value)
            for ev in env_vars:
                try:
                    ev_val = os.environ[ev]
                except KeyError:
                    _logger.exception("Couldn't parse env var '%s' as it hasn't been set", ev)
                    raise
                value = re.sub(env_var_pattern, ev_val, value, count=1)
            return value

        yaml.add_constructor('!envvar', _env_var_constructor, Loader=Config.yaml_loader)

    
class ConfigException(Exception):
    pass
