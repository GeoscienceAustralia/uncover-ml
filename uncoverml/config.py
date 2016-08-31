import logging
from os import path
import glob
import csv
import yaml

from uncoverml import transforms

log = logging.getLogger(__name__)

_imputers = {'mean': transforms.MeanImputer,
             'gaus': transforms.GaussImputer,
             'nn': transforms.NearestNeighboursImputer}
_image_transforms = {'onehot': transforms.OneHotTransform}
_global_transforms = {'centre': transforms.CentreTransform,
                      'standardise': transforms.StandardiseTransform,
                      'whiten': transforms.WhitenTransform}


def _parse_transform_set(transform_dict, imputer_string):
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
                image_transforms.append(_image_transforms[key](**params))
            elif key in _global_transforms:
                global_transforms.append(_global_transforms[key](**params))
    return image_transforms, imputer, global_transforms


class FeatureSetConfig:
    def __init__(self, d):
        self.name = d['name']
        self.type = d['type']

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
                    tifs = [f[0] for f in tifs if len(f) > 0]
                for f in tifs:
                    files.append(path.abspath(f))

        self.files = sorted(files)

        trans_i, im, trans_g = _parse_transform_set(d['transforms'],
                                                    d['imputation'])
        self.transform_set = transforms.ImageTransformSet(trans_i, im, trans_g)


class Config:
    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as f:
            s = yaml.load(f)
        self.name = path.basename(yaml_file).rsplit(".", 1)[0]

        # TODO expose this option when fixed
        if 'patchsize' in s:
            log.info("Patchsize currently fixed at 0 -- ignoring")
        self.patchsize = 0

        self.feature_sets = [FeatureSetConfig(k) for k in s['features']]
        final_transform = s['preprocessing']
        _, im, trans_g = _parse_transform_set(final_transform['transforms'],
                                              final_transform['imputation'])
        self.final_transform = transforms.TransformSet(im, trans_g)
        self.target_file = s['targets']['file']
        self.target_property = s['targets']['property']
        self.algorithm = s['learning']['algorithm']
        self.algorithm_args = s['learning']['arguments']
        memory_fraction = s['memory_fraction']
        self.n_subchunks = max(1, round(1.0 / memory_fraction))
        self.quantiles = s['prediction']['quantiles']

        # TODO pipeline this better
        self.rank_features = 'feature_rank' in s['validation']
        self.cross_validate = False
        for i in s['validation']:
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
