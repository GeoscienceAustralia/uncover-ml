import logging
from os import path
import glob
import csv

import yaml

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
                    tifs = [f[0].strip() for f in tifs if len(f) > 0]
                for f in tifs:
                    files.append(path.abspath(f))

        self.files = sorted(files, key=str.lower)
        n_files = len(self.files)

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
            s = yaml.load(f)
        self.name = path.basename(yaml_file).rsplit(".", 1)[0]

        # TODO expose this option when fixed
        if 'patchsize' in s:
            log.info("Patchsize currently fixed at 0 -- ignoring")
        self.patchsize = 0

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
        self.algorithm = s['learning']['algorithm']
        self.cubist = self.algorithm == 'cubist'
        self.algorithm_args = s['learning']['arguments']
        self.quantiles = s['prediction']['quantiles']

        # TODO pipeline this better
        self.rank_features = False
        self.cross_validate = False
        if s['validation']:
            for i in s['validation']:
                if i == 'feature_rank':
                    self.rank_features = True
                if type(i) is dict and 'k-fold' in i:
                    self.cross_validate = True
                    self.folds = i['k-fold']['folds']
                    self.crossval_seed = i['k-fold']['random_seed']
                    break
        self.output_dir = s['output']['directory']

        if 'clustering' in s:
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
