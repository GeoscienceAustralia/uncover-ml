"""
Tests for CLI commands.
"""
import os
import shutil

import pytest

from uncoverml.config import Config, FeatureSetConfig
from uncoverml import transforms

@pytest.fixture
def config_object(sirsam_rf_conf, sirsam_rf_out):
    if os.path.exists(sirsam_rf_out):
        shutil.rmtree(sirsam_rf_out)
    return Config(sirsam_rf_conf)

def test_attributes_exist(config_object, sirsam_rf_conf, sirsam_rf_out, 
                          sirsam_covariate_paths, sirsam_target_path):
    c = config_object
    assert c.patchsize == 0
    assert c.algorithm == 'multirandomforest'
    assert c.cubist == False
    assert c.multicubist == False
    assert c.multirandomforest == True
    assert c.krige == False
    assert c.algorithm_args['n_estimators'] == 10
    assert c.algorithm_args['target_transform'] == 'log'
    assert c.algorithm_args['forests'] == 20
    assert c.quantiles == 0.95
    assert c.geotif_options == {}
    assert c.outbands == 10
    assert c.thumbnails == 10
    assert c.raw_covariates == os.path.join(sirsam_rf_out, c.name + '_rawcovariates.csv')
    assert c.pk_covariates == os.path.join(sirsam_rf_out, 'features.pk')
    assert c.pk_targets == os.path.join(sirsam_rf_out, 'targets.pk')
    assert c.pk_load == False
    assert not hasattr(c, 'featurevec')
    assert len(c.feature_sets) == 1
    assert c.feature_sets[0].name == 'ordinal_features'
    assert c.feature_sets[0].type == 'ordinal'
    assert sorted(c.feature_sets[0].files) == sorted(sirsam_covariate_paths)
    ts = c.feature_sets[0].transform_set
    assert len(ts.image_transforms) == 0
    assert len(ts.global_transforms) == 2
    assert any(isinstance(t, transforms.CentreTransform) for t in ts.global_transforms)
    assert any(isinstance(t, transforms.StandardiseTransform) 
               for t in ts.global_transforms)
    assert isinstance(ts.imputer, transforms.MeanImputer)
    assert c.final_transform == None
    assert c.target_file == sirsam_target_path
    assert c.target_property == 'Na_log'
    assert c.resample == None
    assert c.mask == None
    assert not hasattr(c, 'retain')
    assert c.lon_lat == None
    assert not hasattr(c, 'lon')
    assert not hasattr(c, 'lat')
    assert c.rank_features == False
    assert c.permutation_importance == False
    assert c.cross_validate == True
    assert c.parallel_validate == True
    assert c.folds == 5
    assert c.crossval_seed == 1
    assert c.output_dir == sirsam_rf_out
    assert c.optimisation == None
    assert c.clustering == False
    assert c.cluster_analysis == False
    assert not hasattr(c, 'clustering_algorithm')
    assert not hasattr(c, 'n_classes')
    assert not hasattr(c, 'oversample_factor')
    assert not hasattr(c, 'semi_supervised')
    assert not hasattr(c, 'class_file')
    assert not hasattr(c, 'class_property')
