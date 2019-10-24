import os
import sys
import glob
import shutil
import csv
import pickle

import pytest

from uncoverml.scripts import uncoverml


class TestLearnCommand:
    """
    Tests the 'learn' command of the UncomverML CLI.
    """
    SIRSAM_RF = 'sirsam_Na_randomforest'

    # Group the outputs of the learn command by filetype to make them easier to test.
    SIRSAM_RF_MODEL = SIRSAM_RF + '.model'

    SIRSAM_RF_JSON_OUTPUT = SIRSAM_RF + '_scores.json'

    SIRSAM_RF_HDF_OUTPUT = SIRSAM_RF + '_results.hdf5'

    SIRSAM_RF_CSV_OUTPUT = [
        SIRSAM_RF + '_results.csv',
        'rawcovariates.csv',
        'rawcovariates_mask.csv'
    ]

    SIRSAM_RF_PICKLE_OUTPUT = [
        'training_data.pk',
        'features.pk',
        'targets.pk'
    ]

    SIRSAM_RF_IMAGE_OUTPUT = [
        '0_Clim_Prescott_LindaGregory.png',
        '0_U_15v1.png',
        '0_U_TH_15.png',
        '0_dem_foc2.png',
        '0_er_depg.png',
        '0_gg_clip.png',
        '0_k_15v5.png',
        '0_tpi_300.png',
        SIRSAM_RF + '_results.png'
    ]

    SIRSAM_RF_OUTPUTS = [SIRSAM_RF_MODEL, SIRSAM_RF_JSON_OUTPUT, SIRSAM_RF_HDF_OUTPUT]
    SIRSAM_RF_OUTPUTS += SIRSAM_RF_CSV_OUTPUT \
                         + SIRSAM_RF_PICKLE_OUTPUT \
                         + SIRSAM_RF_IMAGE_OUTPUT

    @staticmethod
    @pytest.fixture(scope='class', autouse=True)
    def run_sirsam_random_forest_learning(request, sirsam_rf_conf, sirsam_rf_out):
        """
        Run the top level 'learn' command'. Removes generated output on
        completion.
        """
        def finalize():
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
    def test_model_outputs_match(sirsam_rf_out, sirsam_rf_precomp_learn):
        """
        Test that generated model matches precomputed output.
        """
        test_model_path = os.path.join(sirsam_rf_out, TestLearnCommand.SIRSAM_RF_MODEL)
        precomp_model_path = \
            os.path.join(sirsam_rf_precomp_learn, TestLearnCommand.SIRSAM_RF_MODEL)
        with open(test_model_path, 'rb') as tm_pk, open(precomp_model_path, 'rb') as pc_pk:
            test_model, test_config = pickle.load(tm_pk)
            precomp_model, precomp_config = pickle.load(pc_pk)
        assert test_model == precomp_model
        assert test_config == precomp_config
