import tempfile
from pathlib import Path
import yaml
import os
from unittest.mock import patch
from uncoverml.config import _global_transforms, transforms, Config

def test_global_transforms():
    expected_global_transforms = {
        'centre': transforms.CentreTransform, 
        'standardise': transforms.StandardiseTransform, 
        'log': transforms.LogTransform, 
        'sqrt': transforms.SqrtTransform, 
        'whiten': transforms.WhitenTransform
    } 
    assert _global_transforms == expected_global_transforms


def create_config_yaml(content: dict) -> str:
    temp_dir = tempfile.mkdtemp()
    yaml_path = os.path.join(temp_dir, 'test_config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(content, f)
    return yaml_path

def test_config_algorithm_and_flags():
    config_dict = {
        'patchsize': 1,
        'intersected_features': ['f1', 'f2'],
        'prediction': {
            'quantiles': [0.1, 0.5, 0.9],
            'geotif': {'compress': 'lzw'},
            'outbands': 2,
            'thumbnails': 5,
            'prediction_template': 'template.tif'
        },
        'learning': {
            'algorithm': 'cubist',
            'arguments': {}
        },
        'output': {'directory': tempfile.mkdtemp(), 'model': 'dummy.model'},
        'features': [{
            'name': 'test',
            'type': 'ordinal',
            'files': [{'path': __file__}]
        }]
    }
    path = create_config_yaml(config_dict)
    cfg = Config(path)
    assert cfg.algorithm == 'cubist'
    assert cfg.cubist is True
    assert cfg.quantiles == [0.1, 0.5, 0.9]
    assert cfg.outbands == 2
    assert cfg.prediction_template == 'template.tif'
    assert cfg.patchsize == 0
    assert cfg.intersected_features == ['f1', 'f2']

def test_config_pca_variants():
    variants = [
        ({'n_components': 3}, 'n_components', 3),
        ({'keep_fraction': 0.9}, 'keep_fraction', 0.9),
        ({'variation_fraction': 0.8}, 'variation_fraction', 0.8),
    ]
    for whiten, field, expected in variants:
        config_dict = {
            'pca': {},
            'output': {'directory': tempfile.mkdtemp(), 'model': 'dummy.model', 'pca_json': 'out.json'},
            'features': [{
                'name': 'test',
                'type': 'test_type',
                'files': [{'path': __file__}]
            }],
            'preprocessing': {'transforms': [{'whiten': whiten}], 'imputation': None}
        }
        path = create_config_yaml(config_dict)
        cfg = Config(path)
        assert getattr(cfg, field) == expected
        assert cfg.pca is True

@patch('os.path.exists', return_value=True)
def test_configs(mock_path):
    config_dict = {
        'output': {
            'directory': tempfile.mkdtemp(),
            'model': 'model.model',
            'grouped_shapefile': 'grouped.shp',
            'train_shapefile': 'train.shp',
            'oos_shapefile': 'oos.shp'
        },
        'features': [{
            'name': 'pickle_feature',
            'type': 'pickle',
            'files': {
                'covariates': 'covariates.pkl',
                'targets': 'targets.pkl',
                'rawcovariates': 'raw.pkl',
                'rawcovariates_mask': 'mask.pkl',
                'train_data_pk': 'train.pk',
                'featurevec': 'featurevec.npy',
                'plot_covariates': True
            }
        }],
        'learning': {'algorithm': 'cubist', 'arguments': {}, 'optimisation': {}},
        'targets': {
            'file': 'targets.shp',
            'property': 'target_prop',
            'resample': {
                'value': {'method': 'mean'},
                'spatial': {'scale': 100}
            },
            'group_targets': {
                'groups_eps': 10,
                'group_col': 'region'
            },
            'weight_col_name': 'weight',
            'group': {
                'output_group_col_name': 'zone',
                'spatial': {'buffer': 5},
                'fields_to_keep': ['region', 'subregion']
            },
            'split': {
                'group_col_name': 'zone',
                'oos_fraction': 0.25
            }
        },
    }
    config_path = create_config_yaml(config_dict)
    cfg = Config(config_path)

    assert cfg.pickle is True
    assert cfg.pickle_load is True
    assert cfg.pickled_covariates.endswith('covariates.pkl')
    assert cfg.pickled_targets == 'targets.pkl'
    assert cfg.rawcovariates == 'raw.pkl'
    assert cfg.rawcovariates_mask == 'mask.pkl'
    assert cfg.train_data_pk == 'train.pk'
    assert cfg.featurevec.endswith('featurevec.npy')
    assert cfg.plot_covariates is True
    assert cfg.target_file == 'targets.shp'
    assert cfg.target_property == 'target_prop'

@patch('os.path.exists', return_value=True)
def test_feature_ranking( mock_exists):
    config_dict = {
        'output': {'directory': tempfile.mkdtemp(), 'model': 'dummy.model'},
        'features': [{
            'name': 'test',
            'type': 'ordinal',
            'files': [{'path': __file__}]
        }],
        'learning': {'algorithm': 'dummy', 'arguments': {}},
        'validation': [
            'feature_rank',
            'permutation_importance',
            'feature_importance',
            'parallel',
            'shapley',
            {'k-fold': {'folds': 5, 'random_seed': 123}}
        ]
    }

    path = create_config_yaml(config_dict)
    cfg = Config(path)

    assert cfg.rank_features is True
    assert cfg.permutation_importance is True
    assert cfg.feature_importance is True
    assert cfg.parallel_validate is True
    assert cfg.shapley is True
    assert cfg.cross_validate is True
    assert cfg.folds == 5
    assert cfg.crossval_seed == 123
