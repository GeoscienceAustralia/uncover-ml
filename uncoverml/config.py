"""
Handles parsing of the configuration file.
"""
from typing import Optional, List
import logging
import tempfile
import sys
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
        if d['type'] not in ('ordinal', 'categorical'):
            _logger.warning("Feature set type must be ordinal or categorical. "
                            "Unknwon option: '%s'. Type has been set to 'ordinal'.",
                            d['type'])
            self.type = 'ordinal'
        else:
            self.type = d['type']
        is_categorical = d['type'] == 'categorical'

        # get list of all the files
        if 'files' in d:
            self.tabular = False
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
            n_feat = len(self.files)
            _logger.debug("Loaded feature set with files: {self.files}")
        elif 'shapefile' in d:
            self.tabular = True
            self.fields = sorted(d['shapefile']['fields'], key=str.lower)
            n_feat = len(self.fields)
            self.file = d['shapefile']['file']
            self.ndv = d['shapefile'].get('ndv', None)
            _logger.debug(f"Loaded feature set with fields: {self.fields}")

        trans_i, im, trans_g = _parse_transform_set(d['transforms'],
                                                    d['imputation'],
                                                    n_feat)
        
        self.transform_set = transforms.ImageTransformSet(trans_i, im, trans_g, is_categorical)
                                                          

class Config(object):
    """
    Class representing the global configuration of the uncoverml
    scripts.

    This class is *mostly* read-only, but it does also contain the
    Transform objects which have state. In some execution paths, 
    config flags are switched off then back on (e.g. in cross
    validation).

    Along with the YAML file, the init also takes some flags. These
    are set by the top-level CLI scripts and are used to determine
    what parameters to load and what can be ignored.

    All attributes following `output_dir` (located at the bottom of
    init) are undocumented but should be self-explanatory. They are
    full paths to output for different features.

    .. todo::

        Factor out stateful Transform objects.

    Parameters
    ----------
    yaml_file : str
        The path to the yaml config file.
    clustering : bool
        True if clustering.
    learning : bool
        True if learning.
    resampling : bool
        True if resampling.
    predicting : bool
        True if predicting.


    Attributes
    ----------
    name : str
        Name oo the config file.
    algorithm : str
        Name of the model to train. See :ref:`models-page` for available
        models.
    algorithm_args : dict(str, any)
        A dictionary of arguments to pass to selected model. See
        :ref:`models-page` for available arguments to model. Key is 
        the argument name exactly as it appears in model __init__ (this
        dict gets passed as kwargs).
    cubist : bool
        True if cubist algorithm is being used.
    multicubist : bool
        True if multicubist algorithm is being used.
    multirandomforest : bool
        True if multirandomforest algorithm is being used.
    krige : bool
        True if kriging is being used.
    bootstrap : bool
        True if a bootstrapped algorithm is being used.
    clustering : bool
        True if clustering is being performed.
    n_classes : int
        Number of classes to cluster into. Required if clustering.
    oversample_factor : float
        Controls how many candidates are found for cluster 
        initialisation when running kmeans clustering. See
        :func:`~uncoverml.cluster.weighted_starting_candidates`.
        Required when clustering.
    cluster_analysis : bool, optional
        True if analysis should be performed post-clustering. Optional,
        default is False.
    class_file : str or bytes, optional
        Define classes for clustering feature data. Path to shapefile
        that defines class at positions.
    semi_supervised : bool
        True if semi_supervised clustering is being performed (i.e.
        class_file has been provided).
    target_search : bool
        True if `target_search` feature is being used.
    target_search_threshold : float
        Target search threshold, float between 0 and 1. The likelihood
        a training point must surpass to be included in found points.
    target_search_extents : tuple(float, float, float, float)
        A bounding box defining the image area to search for additional
        targets.
    tse_are_pixel_coordinates : bool
        If True, `target_search_extents` are treated as pixel 
        coordinates instead of CRS coordinates.
    extents : tuple(float, float, float, float), optional
        A bounding box defining the area to learn and predict on. Data
        outside these extents gets cropped.
        Optional, if not provided whole image area is used.
    extents_are_pixel_coordinates : bool
        If True, `extents` are treated as pixel coordinates instead of
        CRS coordinates.
    pk_covarates : str or bytes
        Path to where to save pickled covariates, or a pre-existing
        covariate pickle file if loading pickled covariates.
    pk_targets : str or bytes
        Path to where to save pickled targets, or a pre-existing
        target pickle file if loading pickled targets.
    pk_load : bool
        True if both `pk_covariates` and `pk_targets` are provided
        and these paths exist (it's assumed they contain the correct
        pickled data).
    feature_sets : :class:`~uncoverml.config.FeatureSetConfig`
        The provided features as `FeatureSetConfig` objects. These
        contain paths to the feature files and *importantly* the 
        Transform objects which contain statistics used to transform
        the covariates. These Transform objects and contained statistics
        must be maintained across workflow steps (aka CLI commands).
    patchsize : int
        Half-width of the patches that feature data will be chunked
        into. Height/width of each patch is equal to patchsize * 2 + 1.

        .. todo::
            
            Not implemented, defaults to 1.

    target_file : str or bytes
        Path to a shapefile defining the targets to be trained on.
    target_property : str
        Name of the field in the `target_file` to be used as 
        training property.
    target_weight_property : str, optional
        Name of the field in the `target_file` to be used as target
        weights.
    fields_to_write_to_csv : list(str), optional
        List of field names in the `target_file` to be included in
        output table.
    shiftmap_targets : str or bytes, optional
        Path to a shapefile containing targets to generate shiftmap 
        from. This is optional, by default shiftmap will generate
        dummy targets by randomly sampling the target shapefile.
    spatial_resampling_args : dict
        Kwargs for spatial resampling. See :ref:`resampling-section`
        for more details.
    value_resampling_args : dict
        Kwargs for value resampling. See :ref:`resampling-section` for
        more details.
    final_transform : :class:`~uncoverml.transforms.TransformSet`
        Transforms to apply to whole image set after other preprocessing
        has been performed.
    oos_percentage : float, optional
        Float between 0 and 1. The percentage of targets to withhold
        from training to be used in out-of-sample validation.
    oos_shapefile : str or bytes, optional
        Shapefile containing targets to be used in out-of-sample
        validation.
    oos_property : str
        Name of the property in `oos_shapefile` to be used in 
        validation. Only required if an OOS shapefile is provided.
    out_of_sample_validation : bool
        True if out of sample validation is to be performed.
    rank_features : bool, optional
        True if 'feature_ranking' is True in 'validation' block
        of the config. Turns on feature ranking. Default is False.
    permutation_importance : bool
        True if 'permutation_importance' is True in 'validation'
        block of the config. Turns on permutation importance.  
        Default is False.
    parallel_validate : bool, optional
        True if 'parallel' is present in 'k-fold' block of
        config. Turns on parallel k-fold cross validation. Default
        is False.
    cross_validate : bool, optional
        True if 'k-fold' block is present in 'validation' block of 
        config. Turns on k-fold cross validation.
    folds : int
        The number of folds to split dataset into for cross validation.
        Required if :attr:`~cross_validate` is True.
    crossval_seed : int
        Seed for random sorting of folds for cross validation. Required
        if :attr:`~cross_validate` is True.
    optimisation : dict
        Dictionary of optimisation arguments. 
        See :ref:`optimisation-section` for details.
    geotiff_options : dict, optional
        Optional creation options passed to the geotiff output driver.
        See https://gdal.org/drivers/raster/gtiff.html#creation-options
        for a list of creation options.
    quantiles : float
        Prediction quantile/interval for predicted values.
    outbands : int
        The outbands to write in the prediction output file. Used as
        the 'stop' for a slice taken from list of prediction tags,
        i.e. [0: outbands]. If the resulting slice is greater than
        the number of tags available, then all tags will be selected.
        If no value is provied, then all tags will be selected.

        .. todo::

            Having this as a slice is questionable. Should be 
            simplified.

    thumbnails : int, optional
        Subsampling factor for thumbnails of output images. Default
        is 10.
    bootstrap_predictions : int, optional
        Only applies if a bootstrapped algorithm is being used. This is
        the number of predictions to perform, by default will predict 
        on all sub-models. E.g. if you had a BS algorithm containing
        100 sub-models, you could limit a test prediction to 20 using
        this parameter to speed things up.
    mask : str, optional
        Path to a geotiff file for masking the output prediction 
        map. Only values that have been masked will be predicted.
    retain : int
        Value in the above mask that indicates cell should be retained
        and predicted. Must be provided if a mask is provided.
    lon_lat : dict, optional
        Dictionary containing paths to longitude and latitude grids
        used in kriging.
    output_dir : str
        Path to directory where prediciton map and other outputs
        will be written.
    """
    def __init__(self, yaml_file, clustering=False, learning=False, resampling=False,
                 predicting=False, shiftmap=True):

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
            try:
                s = yaml.load(f, Loader=Config.yaml_loader)
            except UnicodeDecodeError:
                if yaml_file.endswith('.model'):
                    _logger.error("You're attempting to run uncoverml but have provided the "
                                  "'.model' file instead of the '.yaml' config file. The predict "
                                  "now requires the configuration file and not the model. Please "
                                  "try rerunning the command with the configuration file.")
                else:
                    _logger.error("Couldn't parse the yaml file. Ensure you've provided the correct "
                                  "file as config file and that the YAML is valid.")
        self.name = path.basename(yaml_file).rsplit(".", 1)[0]

        if clustering:
            # CLUSTERING BLOCK
            cb = _grp(s, 'clustering', "'clustering' block must be provided when clustering.")
            self.clustering = True
            self.algorithm = cb.get('algorithm', 'kmeans')
            self.n_classes = _grp(cb, 'n_classes', "'n_classes' must be provided when clustering.")
            self.oversample_factor = _grp(cb, 'oversample_factor',
                                          "'oversample_factor' must be provided when clustering.")
            self.cluster_analysis = cb.get('cluster_analysis', False)
            self.class_file = cb.get('file')
            if self.class_file:
                self.class_property = _grp(cb, 'property', "'property' must be provided when "
                                           "providing a file for semisupervised clustering.")
            self.semi_supervised = self.class_file is not None
        elif learning:
            # LEARNING BLOCK
            learn_block = _grp(s, 'learning')
            self.clustering = False
            self.cluster_analysis = False
            self.target_search = learn_block.get('target_search', False)
            self.targetsearch_threshold = learn_block.get('target_search_threshold', 0.8)
            tsexb = learn_block.get('target_search_extents')
            self.targetsearch_extents, self.tse_are_pixel_coordinates = Config.parse_extents(tsexb)
            self.algorithm = _grp(learn_block, 'algorithm',
                                  "'algorithm' must be provided as part of 'learning' block.")
            self.algorithm_args = learn_block.get('arguments', {})
        else:
            self.bootstrap = False
            self.algorithm = None
            self.clustering = False
            self.cluster_analysis = False
            self.target_search = False

        self.set_algo_flags()

        
        # EXTENTS
        exb = s.get('extents')
        self.extents, self.extents_are_pixel_coordinates = Config.parse_extents(exb)

        _logger.debug("loaded crop box %s", self.extents)

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
            self.pk_covariates = None
            self.pk_targets = None
            self.pk_featurevec = None

        # FEATURES BLOCK
        # Todo: fix get_image_spec so features are optional if using pickled data.
        # if not self.pk_load:
        if not resampling:
            _logger.warning("'features' are required even when loading from pickled data - this " 
                            "is a work around for getting image specifications. Needs to be fixed.")
            features = _grp(s, 'features', "'features' block must be provided when not loading "
                            "from pickled data.")
            print("loading features")
            self.feature_sets = [FeatureSetConfig(f) for f in features]
            # Mixing tabular and image features not currently supported
            if any(f.tabular for f in self.feature_sets):
                self.tabular_prediction = True
                if not all(f.tabular for f in self.feature_sets):
                    raise ValueError(
                        "Mixing tabular and image features not currently supported. Ensure "
                        "features are only sourced from 'files' or 'table' but not both.")
            else:
                self.tabular_prediction = False

        # Not yet implemented.
        if 'patchsize' in s:
            _logger.info("Patchsize currently fixed at 0 -- ignoring")
        self.patchsize = 0
        
        # TARGET BLOCK
        if (not predicting and not clustering) or shiftmap:
            tb = _grp(s, 'targets', "'targets' block must be provided when not loading from "
                      "pickled data.")
            self.target_file = _grp(tb, 'file', "'file' needs to be provided when specifying "
                                    "targets.")
            if not path.exists(self.target_file):
                raise FileNotFoundError("Target shapefile provided in config does not exist. Check "
                                        "that the 'file' property of the 'targets' block is correct.")
            self.target_property = _grp(tb, 'property', "'property needs to be provided when "
                                        "specifying targets.")
            self.target_drop_values = tb.get('drop', None)
            self.target_weight_property = tb.get('weight_property')
            self.fields_to_write_to_csv = tb.get('write_to_csv')
            self.shiftmap_targets = tb.get('shiftmap')
            rb = tb.get('resample')
            if rb:
                self.spatial_resampling_args = rb.get('spatial')
                self.value_resampling_args = rb.get('value')
                if not (self.spatial_resampling_args or self.value_resampling_args):
                    raise ValueError("At least one of 'spatial' or 'value' resampling parameters "
                                     "must be provided when resampling.")

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
            oos = vb.get('out_of_sample')
            if oos:
                self.oos_percentage = oos.get('percentage', None)
                self.oos_shapefile = oos.get('shapefile', None)
                self.oos_property = oos.get('property', None)
            self.out_of_sample_validation = oos is not None
            self.rank_features = vb.get('feature_rank', False)
            if self.pk_load and self.rank_features:
                _logger.warning("Feature ranking cannot be performed when loading covariates and "
                                "targets from pickled data.")
                self.rank_features = False
            self.permutation_importance = vb.get('permutation_importance', False)
            kfb = vb.get('k-fold')
            if kfb:
                self.folds = _grp(kfb, 'folds', "'folds' (number of folds) must be specified "
                                  "if k-fold cross validation and/or feature ranking is being used.")
                self.crossval_seed = _grp(kfb, 'random_seed', "'random_seed' must be specified "
                                          "if k-fold cross validation and/or feature ranking is "
                                          "being used.")
                self.parallel_validate = kfb.get('parallel', False)
            elif self.rank_features:
                # Feature ranking requires crossval params. Provide defaults if not available.
                self.folds = 5
                self.crossval_seed = 1
                self.parallel_validate = False
            self.cross_validate = kfb is not None
        else:
            self.rank_features = False
            self.permutation_importance = False
            self.parallel_validate = False
            self.out_of_sample_validation = False
            self.cross_validate = False

        # OPTIMISATION BLOCK
        # Note: optimisation options get parsed in scripts/gridsearch.py
        self.optimisation = s.get('optimisation')

        # PREDICT BLOCK
        if predicting:
            pb = _grp(s, 'prediction', "'prediction' block must be provided.")
            self.geotif_options = pb.get('geotif', {})
            self.quantiles = _grp(pb, 'quantiles', "'quantiles' must be provided as part of "
                                  "prediction block.")
            self.outbands = _grp(pb, 'outbands', "'outbands' must be provided as part of prediction "
                                 "block.")
            self.thumbnails = pb.get('thumbnails', 10)
            self.bootstrap_predictions = pb.get('bootstrap')
            mb = s.get('mask')
            if mb:
                self.mask = mb.get('file') 
                if not os.path.exists(self.mask):
                    raise FileNotFoundError("Mask file provided in config does not exist. Check that "
                                            "the 'file' property of the 'mask' block is correct.")
                self.retain = _grp(mb, 'retain', "'retain' must be provided if providing a "
                                   "prediction mask.")
            else:
                self.mask = None
            
            if self.krige:
                # Todo: don't know if lon/lat is compulsory or not for kriging
                self.lon_lat = s.get('lon_lat')
            else:
                self.lon_lat = None


        # OUTPUT BLOCK
        def _outpath(filename):
            return os.path.join(self.output_dir, self.name + f'{filename}')

        ob = _grp(s, 'output', "'output' block is required.")
        self.output_dir = _grp(ob, 'directory', "'directory' for output is required.")
        self.model_file = ob.get('model', _outpath('.model'))

        if ob.get('plot_feature_ranks', False):
            self.plot_feature_ranks = _outpath('_featureranks.png')
            self.plot_feature_rank_curves = _outpath('_featurerank_curves.png')
        else:
            self.plot_feature_ranks = None
            self.plot_feature_rank = None

        if ob.get('plot_intersection', False):
            self.plot_intersection = _outpath('_intersected.png')
        else:
            self.plot_intersection = None

        if ob.get('plot_real_vs_pred', False):
            self.plot_real_vs_pred = _outpath('_real_vs_pred.png')
            self.plot_residual = _outpath('_residual.png')
        else:
            self.plot_real_vs_pred = None
            self.plot_residual = None

        if ob.get('plot_correlation', False):
            self.plot_correlation = _outpath('_correlation.png')
        else:
            self.plot_correlation = None

        if ob.get('plot_target_scaling', False):
            self.plot_target_scaling = _outpath('_target_scaling.png')
        else:
            self.plot_target_scaling = None
        
        self.raw_covariates = _outpath('_rawcovariates.csv')
        self.raw_covariates_mask = _outpath('_rawcovariates_mask.csv')

        self.feature_ranks_file = _outpath('_featureranks.json')

        self.crossval_scores_file = _outpath('_crossval_scores.json')
        self.crossval_results_file = _outpath('_crossval_results.csv')
        self.crossval_results_plot = _outpath('_crossval_results.png')
        self.oos_scores_file = _outpath('_oos_scores.json')
        self.oos_results_file = _outpath('_oos_results.csv')
        self.oos_targets_file = _outpath('_oos_targets.shp')

        self.dropped_targets_file = _outpath('_dropped_targets.txt')
        self.transformed_targets_file = _outpath('_transformed_targets.csv')

        self.metadata_file = _outpath('_metadata.txt')

        self.optimisation_results_file = _outpath('_optimisation.csv')

        self.prediction_file = _outpath('_{}.tif')
        self.prediction_shapefile = _outpath('_prediction')
        self.prediction_prjfile = _outpath('_prediction.prj')

        self.shiftmap_file = _outpath('_shiftmap_{}.tif')
        self.shiftmap_points = _outpath('_shiftmap_generated_points.csv')

        self.targetsearch_generated_points = _outpath('_targetsearch_generated_points.csv')
        self.targetsearch_likelihood = _outpath('_targetsearch_likelihood.csv')
        self.targetsearch_result_data = _outpath('_targetsearch_result.pk') 

        self.resampled_shapefile_dir = os.path.join(self.output_dir, '{}_resampled')
        
        paths = [self.output_dir, os.path.split(self.model_file)[0]]
        for p in paths:
            if p:
                makedirs(p, exist_ok=True)

        self._tmpdir = None

    @staticmethod
    def parse_extents(exb):
        """
        Validates extents parameters.
        """
        if exb is not None:
            extents = exb.get('xmin'), exb.get('ymin'), exb.get('xmax'), exb.get('ymax')
            if all(x is None for x in extents): 
                _logger.warning("'extents' block was specified but no coordinates or pixel values "
                                "were given. Cropping will not be performed.")

            if (extents[0] and extents[1])  is not None and extents[0] > extents[2]:
                raise ValueError(f"Error in provided crop coordinates: xmin ({extents[0]}) must be less "
                                 f"than xmax ({extents[2]}).")
            elif (extents[2] and extents[3]) is not None and extents[1] > extents[3]:
                raise ValueError(f"Error in provided crop coordinates: ymin ({extents[1]}) must be less "
                                 f"than ymax ({extents[3]}).")
            extents_are_pixel_coordinates = exb.get('pixel_coordinates', False)
            return extents, extents_are_pixel_coordinates
        else:
            return None, None


    @property
    def tmpdir(self):
        """
        Convenience method for creating tmpdir needed by some UncoverML
        functionality.
        """
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp()
        return self._tmpdir

    def set_algo_flags(self):
        """
        Convenience method for setting boolean flags based on the
        algorithm being used.
        """
        # Set flags based on algorithm being used - these control
        # some special behaviours in the code.
        self.cubist = self.algorithm == 'cubist'
        self.multicubist = self.algorithm == 'multicubist'
        self.multirandomforest = self.algorithm == 'multirandomforest'
        self.krige = self.algorithm == 'krige'
        if self.algorithm is not None:
            self.bootstrap = self.algorithm.startswith('bootstrap')

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
                    if not ev_val:
                        raise ValueError   
                except (KeyError, ValueError):
                    _logger.exception("Couldn't parse environment var '%s' as it hasn't been set. "
                                      "Set the variable or remove it from the config file.", ev)
                    sys.exit(1)
                value = re.sub(env_var_pattern, ev_val, value, count=1)
            return value

        yaml.add_constructor('!envvar', _env_var_constructor, Loader=Config.yaml_loader)

    
class ConfigException(Exception):
    pass
