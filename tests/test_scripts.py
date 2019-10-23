import os
import sys
import glob
import shutil

import pytest

from uncoverml.scripts import uncoverml


SIRSAM_RF_OUTPUTS = [
    'features.pk',
    'targets.pk',
    'rawcovariates.csv',
    'rawcovariates_mask.csv',
    'sirsam_Na.model',
    'sirsam_Na_results.csv',
    'sirsam_Na_results.hdf5',
    'sirsam_Na_results.png',
    'sirsam_Na_scores.json',
    'training_data.pk'
]

# Until we refactor output file directories, this separate list of top-level directory outputs
# is required.
SIRSAM_RF_TOPLEVEL_OUTPUTS = [
    '0_Clim_Prescott_LindaGregory.png',
    '0_U_15v1.png',
    '0_U_TH_15.png',
    '0_dem_foc2.png',
    '0_er_depg.png',
    '0_gg_clip.png',
    '0_k_15v5.png',
    '0_tpi_300.png'
]

@pytest.fixture(scope='module', autouse=True)
def run_sirsam_random_forest_learning(sirsam_rf_conf):
    uncoverml.learn([sirsam_rf_conf, '-p', 20])

@pytest.fixture(params=SIRSAM_RF_OUTPUTS)
def sirsam_rf_output(request, sirsam_rf_out):
    return os.path.join(sirsam_rf_out, request.param)

# Todo: consolidate with above fixture after output refactor
@pytest.fixture(params=SIRSAM_RF_TOPLEVEL_OUTPUTS)
def sirsam_rf_toplevel_output(request, sirsam_rf):
    return os.path.join(sirsam_rf, request.param)

def test_output_exists(sirsam_rf_output):
    assert os.path.exists(sirsam_rf_output)

# Todo: consolidate with above test after output refactor
def test_toplevel_output_exists(sirsam_rf_toplevel_output):
    assert os.path.exists(sirsam_rf_toplevel_output)





