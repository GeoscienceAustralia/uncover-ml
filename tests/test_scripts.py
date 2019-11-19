"""
Tests for CLI commands.
"""
import os
import shutil
import pickle
import json

import pytest
import numpy as np

from uncoverml.scripts import uncoverml


class TestLearnCommand:
    """
    Tests the 'learn' command of the UncomverML CLI.
    """
    SIRSAM_RF = 'sirsam_Na_randomforest'

    # Group the outputs of the learn command by filetype to make them easier to test.
    SIRSAM_RF_JSON_OUTPUT = [
        SIRSAM_RF + '_crossval_scores.json',
        SIRSAM_RF + '_featureranks.json'
    ]

    SIRSAM_RF_HDF_OUTPUT = SIRSAM_RF + '_crossval_scores.hdf5'

    SIRSAM_RF_FEATURE_DATA = 'features.pk'
    SIRSAM_RF_TARGET_DATA = 'targets.pk'
    SIRSAM_RF_MODEL = SIRSAM_RF + '.model'

    SIRSAM_RF_CSV_OUTPUT = [
        SIRSAM_RF + '_crossval_results.csv',
        SIRSAM_RF + '_rawcovariates.csv',
        SIRSAM_RF + '_rawcovariates_mask.csv'
    ]

    SIRSAM_RF_IMAGE_OUTPUT = [
        SIRSAM_RF + '_featureranks.png',
        SIRSAM_RF + '_featurerank_curves.png',
        SIRSAM_RF + '_intersected.png',
        SIRSAM_RF + '_correlation.png',
        SIRSAM_RF + '_target_scaling.png'
    ]
     
   
    SIRSAM_RF_OUTPUTS = [SIRSAM_RF_HDF_OUTPUT, SIRSAM_RF_FEATURE_DATA, SIRSAM_RF_TARGET_DATA, 
                         SIRSAM_RF_MODEL]
    SIRSAM_RF_OUTPUTS += SIRSAM_RF_CSV_OUTPUT + SIRSAM_RF_IMAGE_OUTPUT + SIRSAM_RF_JSON_OUTPUT

    @staticmethod
    @pytest.fixture(scope='class', autouse=True)
    def run_sirsam_random_forest_learning(request, sirsam_rf_conf, sirsam_rf_out):
        """
        Run the top level 'learn' command'. Removes generated output on
        completion.
        """
        def finalize():
            if os.path.exists(sirsam_rf_out):
            	shutil.rmtree(sirsam_rf_out)

        request.addfinalizer(finalize)

        try:
            return uncoverml.learn([sirsam_rf_conf, '-p', 20])
        # Catch SystemExit as it gets raised by Click on command completion
        except SystemExit:
            pass
    
    @staticmethod
    @pytest.fixture(params=SIRSAM_RF_OUTPUTS)
    def sirsam_rf_output(request, sirsam_rf_out):
        return os.path.join(sirsam_rf_out, request.param)
    
    @staticmethod
    def test_output_exists(sirsam_rf_output):
        """
        Test that excepted outputs of 'learn' command exist after
        running.
        """
        assert os.path.exists(sirsam_rf_output)

    @staticmethod
    @pytest.fixture(params=SIRSAM_RF_CSV_OUTPUT)
    def sirsam_rf_csv_outputs(request, sirsam_rf_out, sirsam_rf_precomp_learn):
        return (
            os.path.join(sirsam_rf_out, request.param),
            os.path.join(sirsam_rf_precomp_learn,  request.param)
        )

    @staticmethod
    def test_csv_outputs_match(sirsam_rf_csv_outputs):
        """
        Test that CSV covariate info matches precomputed output.
        """
        with open(sirsam_rf_csv_outputs[0]) as test, \
                open(sirsam_rf_csv_outputs[1]) as precomp:
            test_lines = test.readlines()
            precomp_lines = precomp.readlines()
        assert test_lines == precomp_lines


    @staticmethod
    @pytest.fixture(params=SIRSAM_RF_JSON_OUTPUT)
    def sirsam_rf_json_outputs(request, sirsam_rf_out, sirsam_rf_precomp_learn):
        return (
            os.path.join(sirsam_rf_out, request.param),
            os.path.join(sirsam_rf_precomp_learn, request.param)
        )

    @staticmethod
    def test_json_outputs_match(sirsam_rf_json_outputs):
        """
        Test that JSON scores matches precomputed output.
        """
        with open(sirsam_rf_json_outputs[0]) as tf, open(sirsam_rf_json_outputs[1]) as pf:
            test_json = json.load(tf)
            precomp_json = json.load(pf)
        assert test_json == precomp_json

    @classmethod
    def test_model_outputs_match(cls, sirsam_rf_out, sirsam_rf_precomp_learn):
        """
        Test that generated model matches precomputed model.
        """
        t_dict = _unpickle(os.path.join(sirsam_rf_out, cls.SIRSAM_RF_MODEL))
        p_dict = _unpickle(os.path.join(sirsam_rf_precomp_learn, cls.SIRSAM_RF_MODEL))
        assert t_dict['model'] == p_dict['model']
        assert t_dict['config'] == p_dict['config']

    @classmethod
    def test_training_data_matches(cls, sirsam_rf_out, sirsam_rf_precomp_learn):
        """
        Test that pickled training data matches precomputed output.
        """
        t_image_chunk_sets, t_transform_sets, t_targets = \
            _unpickle(os.path.join(sirsam_rf_out, cls.SIRSAM_RF_TRAINING_DATA))
        p_image_chunk_sets, p_transform_sets, p_targets = \
            _unpickle(os.path.join(sirsam_rf_precomp_learn, cls.SIRSAM_RF_TRAINING_DATA))
        assert t_image_chunk_sets == p_image_chunk_sets
        assert t_transform_sets == p_transform_sets
        assert t_targets == p_targets

    @classmethod
    def test_pickled_features_match(cls, sirsam_rf_out, sirsam_rf_precomp_learn):
        """
        Test that pickled features match precomputed output.
        """
        t_features = _unpickle(os.path.join(sirsam_rf_out, cls.SIRSAM_RF_FEATURE_DATA))
        p_features = _unpickle(os.path.join(sirsam_rf_precomp_learn, cls.SIRSAM_RF_FEATURE_DATA))
        assert np.array_equal(t_features, p_features)

    @classmethod
    def test_pickled_targets_match(cls, sirsam_rf_out, sirsam_rf_precomp_learn):
        """
        Test that pickled targets match precomputed output.
        """
        t_targets = _unpickle(os.path.join(sirsam_rf_out, cls.SIRSAM_RF_TARGET_DATA))
        p_targets = _unpickle(os.path.join(sirsam_rf_precomp_learn, cls.SIRSAM_RF_TARGET_DATA))
        assert t_targets == p_targets

def _unpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

@pytest.mark.predict
class TestPredictCommand:
    SIRSAM_RF = 'sirsam_Na_randomforest'
    
    SIRSAM_PREDICTION_MAPS = [
        SIRSAM_RF + '_lower_quantile.tif',
        SIRSAM_RF + '_lower_quantile_thumbnail.tif',
        SIRSAM_RF + '_prediction.tif',
        SIRSAM_RF + '_prediction_thumbnail.tif',
        SIRSAM_RF + '_upper_quantile.tif',
        SIRSAM_RF + '_upper_quantile_thumbnail.tif',
        SIRSAM_RF + '_variance.tif',
        SIRSAM_RF + '_variance_thumbnail.tif'
    ]
    
    SIRSAM_RF_IMAGE_OUTPUT = [
        SIRSAM_RF + '_real_vs_pred.png',
        SIRSAM_RF + '_residual.png'
    ]
        
    SIRSAM_RF_MF_METADATA = SIRSAM_RF + '_metadata.txt'

    @staticmethod
    @pytest.fixture(scope='class', autouse=True)
    def run_sirsam_random_forest_prediction(request, sirsam_rf_out, sirsam_rf_conf, 
                                            sirsam_rf_precomp_learn):
        """
        Run the top level 'predict' command'. Removes generated output on
        completion.
        """
        def finalize():
            if os.path.exists(sirsam_rf_out):
                shutil.rmtree(sirsam_rf_out)

        request.addfinalizer(finalize)

        # Copy precomputed files from learn step to the output directory
        shutil.copytree(sirsam_rf_precomp_learn, sirsam_rf_out)

        try:
            return uncoverml.predict([sirsam_rf_conf, '-p', 30])
        # Catch SystemExit as it gets raised by Click on command completion
        except SystemExit:
            pass
    
    @staticmethod
    @pytest.fixture(params=SIRSAM_PREDICTION_MAPS + [SIRSAM_RF_MF_METADATA] + SIRSAM_RF_IMAGE_OUTPUT)
    def sirsam_rf_output(request, sirsam_rf_out):
        return os.path.join(sirsam_rf_out, request.param)
    
    @staticmethod
    def test_output_exists(sirsam_rf_output):
        """
        Test that excepted outputs of 'predict' command exist after
        running.
        """
        assert os.path.exists(sirsam_rf_output)

