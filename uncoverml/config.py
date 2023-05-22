import logging
from os import path
from os import makedirs
import glob
import csv
import yaml
from pathlib import Path

from uncoverml import transforms

log = logging.getLogger(__name__)

"""The strings associated with each imputation option
"""
_imputers = {'mean': transforms.MeanImputer,
             'gaus': transforms.GaussImputer,
             'nn': transforms.NearestNeighboursImputer}

"""These transforms operate individually on each image before concatenation
"""
_image_transforms = {'onehot': transforms.OneHotTransform,
                     'randomhot': transforms.RandomHotTransform}

"""Post-concatenation transforms: operate on whole data vector
"""
_global_transforms = {'centre': transforms.CentreTransform,
                      'standardise': transforms.StandardiseTransform,
                      'log': transforms.LogTransform,
                      'sqrt': transforms.SqrtTransform,
                      'whiten': transforms.WhitenTransform}


def _parse_transform_set(transform_dict, imputer_string, n_images=None):
    """Parse a dictionary read from yaml into a TransformSet object

    Parameters
    ----------
    transform_dict : dictionary
        The dictionary as read from the yaml config file containing config
        key-value pairs
    imputer_string : string
        The name of the imputer (could be None)
    n_images : int > 0
        The number of images being read in. Required because we need to create
        a new image transform for each image

    Returns
    -------
    image_transforms : list
        A list of image Transform objects
    imputer : Imputer
        An Imputer object
    global_transforms : list
        A list of global Transform objects
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


class FeatureSetConfig:
    """Config class representing a 'feature set' in the config file

    Parameters
    ----------
    d : dictionary
        The section of the yaml file for a feature set
    """
    def __init__(self, d):
        self.name = d['name']
        self.type = d['type']
        if d['type'] not in {'ordinal', 'categorical'}:
            log.warning("Feature set type must be ordinal or categorical: "
                        "Unknown option "
                        "{} (assuming ordinal)".format(d['type']))
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
        if 'transforms' not in d:
            d['transforms'] = None

        if 'imputation' not in d:
            d['imputation'] = None

        trans_i, im, trans_g = _parse_transform_set(d['transforms'],
                                                    d['imputation'],
                                                    n_files)
        self.transform_set = transforms.ImageTransformSet(trans_i, im, trans_g,
                                                          is_categorical)


class Config:
    """Class representing the global configuration of the uncoverml scripts

    This class is *mostly* read-only, but it does also contain the Transform
    objects which have state. TODO: separate these out!

    Parameters
    ----------
    yaml_file : string
        The path to the yaml config file. For details on the yaml schema
        see the uncoverml documentation
    """
    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as f:
            s = yaml.safe_load(f)
        self.name = path.basename(yaml_file).rsplit(".", 1)[0]

        # TODO expose this option when fixed
        if 'patchsize' in s:
            log.info("Patchsize currently fixed at 0 -- ignoring")
        self.patchsize = 0

        if 'intersected_features' in s:
            self.intersected_features = s['intersected_features']
        else:
            self.intersected_features = None

        if 'learning' in s:
            self.algorithm = s['learning']['algorithm']
            self.algorithm_args = s['learning']['arguments']
        else:
            self.algorithm = None

        self.cubist = self.algorithm == 'cubist'
        self.multicubist = self.algorithm == 'multicubist'
        self.multirandomforest = self.algorithm == 'multirandomforest'
        self.multisvr = self.algorithm == 'svrmulti'
        self.krige = self.algorithm == 'krige'

        if 'prediction' in s:
            prediction = s["prediction"]
            self.quantiles = prediction['quantiles']
            self.geotif_options = prediction['geotif'] if 'geotif' in prediction else {}
            self.outbands = None
            if 'outbands' in prediction:
                self.outbands = prediction['outbands']
            self.thumbnails = prediction['thumbnails'] \
                if 'thumbnails' in s['prediction'] else 10

            self.prediction_template = prediction["prediction_template"] \
                if "prediction_template" in prediction else None

        self.is_prediction = False

        if 'features' in s:
            self.pickle = any(True for d in s['features'] if d['type'] == 'pickle')
        else:
            self.pickle = False

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
                    if 'plot_covariates' in d['files']:
                        self.plot_covariates = d['files']['plot_covariates']
                    else:
                        self.plot_covariates = False
                    s['features'].pop(n)  # pop `pickle` features
        else:
            self.pickle_load = False


        if 'features' in s:
            self.feature_sets = [FeatureSetConfig(k) for k in s['features']]

        if 'preprocessing' in s:
            final_transform = s['preprocessing']
            if 'transforms' not in final_transform:
                final_transform['transforms'] = None
            if 'imputation' not in final_transform:
                final_transform['imputation'] = None
            _, im, trans_g = _parse_transform_set(
                final_transform['transforms'], final_transform['imputation'])
            self.final_transform = transforms.TransformSet(im, trans_g)
        else:
            self.final_transform = None

        self.output_dir = s['output']['directory']
        # create output dir if does not exist
        makedirs(self.output_dir, exist_ok=True)

        if 'oos_validation' in s:
            self.oos_validation_file = s['oos_validation']['file']
            self.oos_validation_property = s['oos_validation']['property']

        if 'targets' in s:
            self.target_file = s['targets']['file']
            if 'property' in s['targets']:
                self.target_property = s['targets']['property']
            self.resample = None
            if 'resample' in s['targets']:
                self.resample = s['targets']['resample']
                self.value_resampling_args = None
                self.spatial_resampling_args = None
                if 'value' in s['targets']['resample']:
                    self.value_resampling_args = s['targets']['resample']['value']
                if 'spatial' in s['targets']['resample']:
                    self.spatial_resampling_args = s['targets']['resample']['spatial']
                if self.value_resampling_args is None and self.spatial_resampling_args is None:
                    raise ValueError("provide at least one of value or spatial args for resampling")

            if 'group_targets' in s['targets']:
                self.group_targets = True
                self.groups_eps = s['targets']['group_targets']['groups_eps']
                if 'group_col' in s['targets']['group_targets']:
                    self.group_col = s['targets']['group_targets']['group_col']
                else:
                    self.group_col = None
                self.target_groups_file = path.join(self.output_dir, 'target_groups.jpg')
            else:
                self.group_targets = False
            self.target_groups_file = path.join(self.output_dir, 'target_groups.jpg')
            if 'weight_col_name' in s['targets']:
                self.weighted_model = True
                self.weight_col_name = s['targets']['weight_col_name']
            else:
                self.weighted_model = False

            if 'group' in s['targets']:
                self.output_group_col_name = s['targets']['group']['output_group_col_name']
                if 'spatial' in s['targets']['group']:
                    self.spatial_grouping_args = s['targets']['group']['spatial']
                else:
                    self.spatial_grouping_args = {}
                if 'fields_to_keep' in s['targets']['group']:
                    self.grouping_fields_to_keep = s['targets']['group']['fields_to_keep']
                else:
                    self.grouping_fields_to_keep = []
                self.grouped_output = Path(self.output_dir).joinpath(s['output']['grouped_shapefile'])

            if 'split' in s['targets']:
                self.split_group_col_name = s['targets']['split']['group_col_name']
                self.split_oos_fraction = s['targets']['split']['oos_fraction']
                self.train_shapefile = Path(self.output_dir).joinpath(s['output']['train_shapefile'])
                self.oos_shapefile = Path(self.output_dir).joinpath(s['output']['oos_shapefile'])
            self.resampled_output = Path(self.output_dir).joinpath(Path(self.target_file).stem + '_resampled.shp')

        self.mask = None
        if 'mask' in s:
            self.mask = s['mask']['file']
            self.retain = s['mask']['retain']  # mask areas that are predicted

        if 'pca' in s:
            self.pca = True
            preprocessing_transforms = s['preprocessing']['transforms']
            if 'n_components' in preprocessing_transforms[0]['whiten']:
                self.n_components = preprocessing_transforms[0]['whiten']['n_components']
            elif 'keep_fraction' in preprocessing_transforms[0]['whiten']:
                self.keep_fraction = preprocessing_transforms[0]['whiten']['keep_fraction']
            elif 'variation_fraction' in preprocessing_transforms[0]['whiten']:
                self.variation_fraction = preprocessing_transforms[0]['whiten']['variation_fraction']
            else:
                self.n_components = None
            if 'geotif' not in s['pca']:
                tif_opts = {}
            else:
                tif_opts = s['pca']['geotif'] if s['pca']['geotif'] is not None else {}
            self.geotif_options = tif_opts
            self.pca_json = path.join(self.output_dir, s['output']['pca_json'])
        else:
            self.pca = False

        self.lon_lat = False
        if 'lon_lat' in s:
            self.lon_lat = True
            self.lat = s['lon_lat']['lat']
            self.lon = s['lon_lat']['lon']

        # TODO pipeline this better
        self.rank_features = False
        self.permutation_importance = False
        self.feature_importance = False
        self.cross_validate = False
        self.parallel_validate = False
        if 'validation' in s:
            for i in s['validation']:
                if i == 'feature_rank':
                    self.rank_features = True
                if i == 'permutation_importance':
                    self.permutation_importance = True
                if i == 'feature_importance':
                    self.feature_importance = True
                if i == 'parallel':
                    self.parallel_validate = True
                if type(i) is dict and 'k-fold' in i:
                    self.cross_validate = True
                    self.folds = i['k-fold']['folds']
                    self.crossval_seed = i['k-fold']['random_seed']
                    break

        if self.rank_features and self.pickle_load:
            self.pickle_load = False
            log.info('Feature ranking does not work with '
                     'pickled files. Pickled files will not be used. '
                     'All covariates will be intersected.')

        self.optimised_model = False
        if 'learning' in s:
            self.hpopt = False
            self.skopt = False
            if 'optimisation' in s['learning']:
                if 'searchcv_params' in s['learning']['optimisation']:
                    self.opt_searchcv_params = s['learning']['optimisation']['searchcv_params']
                    self.opt_params_space = s['learning']['optimisation']['params_space']
                    self.skopt = True
                if 'hyperopt_params' in s['learning']['optimisation']:
                    self.hyperopt_params = s['learning']['optimisation']['hyperopt_params']
                    self.hp_params_space = s['learning']['optimisation']['hp_params_space']
                    self.hpopt = True

                if self.skopt and self.hpopt:
                    raise ConfigException("Only one of searchcv_params or hyperopt_params can be specified")

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

        output_model = s['output']['model'] if 'model' in s['output'] \
            else self.name + ('.cluster' if self.clustering else '.model')

        self.model_file = Path(self.output_dir).joinpath(output_model)
        self.optimisation_output_skopt = Path(self.output_dir).joinpath(self.name + '_optimisation_skopt.csv')
        self.optimisation_output_hpopt = Path(self.output_dir).joinpath(self.name + '_optimisation_hpopt.csv')
        self.optimised_model_params = Path(self.output_dir).joinpath(self.name + "_optimised_params.json")
        self.optimised_model_file = Path(self.output_dir).joinpath(self.name + "_optimised.model")
        self.outfile_scores = Path(self.output_dir).joinpath(self.name + "_optimised_scores.json")
        self.optimised_model_scores = Path(self.output_dir).joinpath(self.name + "_optimised_scores.json")


class ConfigException(Exception):
    pass
