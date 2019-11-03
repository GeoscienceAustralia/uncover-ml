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

    

def _parse_transform_set(transform_dict: dict, imputer_string: str, n_images: int) -> tuple:
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
    if imputer_string in _imputers:
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
    config_file : str
        Absolute path to the config yaml file.
    name : str
        Name fo the config file.
    patchsize : int
        Half-width of the patches that feature data will be chunked
        into. Height/width of each patch is equal to patchsize * 2 + 1.
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
    rawcovariates : str
        Path to file for saving intersected features and targets
        as CSV before processing is performed (hence 'raw'). If not
        present then saving will not occur.
    rawcovariates_mask : str
        Corresponding mask for the raw covariates in CSV format. Must
        be provided if raw covariates is provided.
    train_data_pk  : str
        Path to file containing pickled training data. If file exists,
        image chunk sets, transform sets and targets will be loaded
        from the pickle file. Training data will be pickled and saved
        to the specified file after its creation.
    pickled_covariates : str
        Path to pickle file containing intersection of targets and
        covariates. If :attr:`~pikcle_load` is True, then this file
        will be loaded and used as covariates for learning. If
        :attr:`~pickle_load` is False, then covariates will be dumped
        to this file after they have been created from the feature
        sets specified in the configuration.
    pickled_targets : str
        Path to pickle file containing targets. If :attr:`~pickle_load`
        is True, then this file will be loaded and used as targets
        for learning. If :attr:`~pickle_load` is False, then targets
        will be dumped to this file after they have been created from
        the target file specified in the config.
    pickle_load : bool
        True if :attr:`~pickle` is True and existent files are provided
        for :attr:`~pickled_covariates` and :attr:`~pickled_targets`.
        If True, then covariates and targets will be loaded from the
        pickle files. If False, the created targets and covariates
        will be dumped if respective file paths are provided. Will
        also be set to False if :attr:`~rank_features` is True as
        feature ranking is not compatible with pickled data.
    featurevec : str
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

            Doesn't appear to be implemented
    
    mask : str
        Path to a geotiff file for masking the output prediction 
        map. Only values that have been masked will be predicted.
    retain : int
        Value in the above mask that indicates cell should be retained
        and predicted.
    lon_lat : bool
        True if 'lon_lat' block is present.
    lon : str
        Path to geotiff file containing latitiude grid for Kriging.
    lat : str
        Path to geotiff file containing longitude grid for Kriging.
    rank_features : bool
        True if 'feature_ranking' is present in 'validation' block
        of the config. Turns on feature ranking.
    permutation_importance : bool
        True if 'permutation_importance' is present in 'validation'
        block of the config. Turns on permutation importance.  
    parallel_validate : bool
        True if 'parallel' is present in 'validation' block of
        config. Turns on parallel k-fold cross validation.
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
    optimisation : dict
        Dictionary of optimisation arguments.
    optimisation_output : str
        Filname for output of optimisation.
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
    def __init__(self, yaml_file: str):
        Config._configure_pyyaml()
        self.config_yaml = yaml_file
        with open(yaml_file, 'r') as f:
            s = yaml.load(f, Loader=Config.yaml_loader)
        self.name = path.basename(yaml_file).rsplit(".", 1)[0]

        # TODO expose this option when fixed
        if 'patchsize' in s:
            _logger.info("Patchsize currently fixed at 0 -- ignoring")
        self.patchsize = 0

        self.algorithm = s['learning']['algorithm']
        self.cubist = self.algorithm == 'cubist'
        self.multicubist = self.algorithm == 'multicubist'
        self.multirandomforest = self.algorithm == 'multirandomforest'
        self.krige = self.algorithm == 'krige'
        self.algorithm_args = s['learning']['arguments']
        self.quantiles = s['prediction']['quantiles']

        self.geotif_options = s['prediction']['geotif'] if 'geotif' in \
            s['prediction'] else {}

        self.outbands = None
        if 'outbands' in s['prediction']:
            self.outbands = s['prediction']['outbands']
        self.thumbnails = s['prediction']['thumbnails'] \
            if 'thumbnails' in s['prediction'] else 10

        self.pickle = any(d['type'] == 'pickle' for d in s['features'])

        self.rawcovariates = False
        self.train_data_pk = False
        if self.pickle:
            self.pickle_load = True
            for n, d in enumerate(s['features']):
                if d['type'] == 'pickle':
                    if 'covariates' in d['files']:
                        self.pickled_covariates = \
                            path.abspath(d['files']['covariates'])
                    if 'targets' in d['files']:
                        self.pickled_targets = d['files']['targets']
                    if 'rawcovariates' in d['files']:
                        self.rawcovariates = d['files']['rawcovariates']
                        self.rawcovariates_mask = \
                            d['files']['rawcovariates_mask']
                    if 'train_data_pk' in d['files']:
                        self.train_data_pk = d['files']['train_data_pk']
                    if not (path.exists(d['files']['covariates'])
                            and path.exists(d['files']['targets'])):
                        self.pickle_load = False
                    if self.cubist or self.multicubist:
                        if 'featurevec' in d['files']:
                            self.featurevec = \
                                path.abspath(d['files']['featurevec'])
                        if not path.exists(d['files']['featurevec']):
                            self.pickle_load = False
                    self.plot_covariates = d['files'].get('plot_covariates')
                    s['features'].pop(n)  # pop `pickle` features
        else:
            self.pickle_load = False

        if not self.pickle_load:
            _logger.info('One or both pickled files were not '
                     'found. All targets will be intersected.')

        self.feature_sets = [FeatureSetConfig(k) for k in s['features']]

        if 'preprocessing' in s:
            final_transform = s['preprocessing']
            _, im, trans_g = _parse_transform_set(
                final_transform['transforms'], final_transform['imputation'])
            self.final_transform = transforms.TransformSet(im, trans_g)
        else:
            self.final_transform = None

        self.target_file = s['targets']['file']
        self.target_property = s['targets']['property']

        self.resample = None

        if 'resample' in s['targets']:
            self.resample = s['targets']['resample']

        self.mask = None
        if 'mask' in s:
            self.mask = s['mask']['file']
            self.retain = s['mask']['retain']  # mask areas that are predicted

        self.lon_lat = False
        if 'lon_lat' in s:
            self.lon_lat = True
            self.lat = s['lon_lat']['lat']
            self.lon = s['lon_lat']['lon']

        # TODO pipeline this better
        self.rank_features = False
        self.permutation_importance = False
        self.cross_validate = False
        self.parallel_validate = False
        if s['validation']:
            for i in s['validation']:
                if i == 'feature_rank':
                    self.rank_features = True
                if i == 'permutation_importance':
                    self.permutation_importance = True
                if i == 'parallel':
                    self.parallel_validate = True
                if type(i) is dict and 'k-fold' in i:
                    self.cross_validate = True
                    self.folds = i['k-fold']['folds']
                    self.crossval_seed = i['k-fold']['random_seed']
                    break

        if self.rank_features and self.pickle_load:
            self.pickle_load = False
            _logger.info('Feature ranking does not work with '
                     'pickled files. Pickled files will not be used. '
                     'All covariates will be intersected.')

        # OUTPUT BLOCK
        output_dict = s['output']
        self.output_dir = output_dict['directory']
        self.model_file = \
            output_dict.get('model', os.path.join(
                self.output_dir, self.name + '_' + self.algorithm + '.model'))
        makedirs(self.output_dir, exist_ok=True)

        if 'optimisation' in s:
            self.optimisation = s['optimisation']
            if 'optimisation_output' in self.optimisation:
                self.optimisation_output = \
                    self.optimisation['optimisation_output']

        self.cluster_analysis = False
        self.clustering = False
        if 'clustering' in s:
            self.clustering = True
            self.clustering_algorithm = s['clustering']['algorithm']
            cluster_args = s['clustering']['arguments']
            self.n_classes = cluster_args['n_classes']
            self.oversample_factor = cluster_args['oversample_factor']
            if 'file' in s['clustering'] and s['clustering']['file']:
                self.semi_supervised = True
                self.class_file = s['clustering']['file']
                self.class_property = s['clustering']['property']
            else:
                self.semi_supervised = False
            if 'cluster_analysis' in s['clustering']:
                self.cluster_analysis = s['clustering']['cluster_analysis']

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
