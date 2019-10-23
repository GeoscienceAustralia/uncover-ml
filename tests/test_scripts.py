import os
import sys
import glob
import shutil

import pytest

from uncoverml.scripts import uncoverml


SIRSAM_RF_LEARN_OUTPUTS = [
    'features.pk',
    'targets.pk',
    'rawcovariates.csv',
    'rawcovariates_mask.csv',
    'sirsam_Na_randomforest.model',
    'sirsam_Na_randomforest_results.csv',
    'sirsam_Na_randomforest_results.hdf5',
    'sirsam_Na_randomforest_results.png',
    'sirsam_Na_randomforest_scores.json',
    'training_data.pk',
    '0_Clim_Prescott_LindaGregory.png',
    '0_U_15v1.png',
    '0_U_TH_15.png',
    '0_dem_foc2.png',
    '0_er_depg.png',
    '0_gg_clip.png',
    '0_k_15v5.png',
    '0_tpi_300.png'
]

class TestLearnCommand:
    @pytest.fixture(scope='class', autouse=True)
    def run_sirsam_random_forest_learning(self, request, sirsam_rf_conf, sirsam_rf_out):
        def finalize():
            shutil.rmtree(sirsam_rf_out)

        request.addfinalizer(finalize)

        try:
            return uncoverml.learn([sirsam_rf_conf, '-p', 20])
        # Catch SystemExit as it gets raised by Click on command competion
        except SystemExit:
            pass
        
    @pytest.fixture(params=SIRSAM_RF_LEARN_OUTPUTS)
    def sirsam_rf_output(self, request, sirsam_rf_out):
        return os.path.join(sirsam_rf_out, request.param)

    def test_output_exists(self, sirsam_rf_output):
        print(f"testing that '{sirsam_rf_output}' exists...")
        assert os.path.exists(sirsam_rf_output)
