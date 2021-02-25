"""
Integration tests for CLI commands.
"""
import os
import shutil
import pickle
import json
import subprocess
import pytest
import rasterio
import numpy as np

import uncoverml.scripts
from uncoverml.models import SVRTransformed

SIRSAM_RF = 'sirsam_Na_randomforest'


class TestBootstrap:
    output_files = [
        'bootstrapping_lower_quantile_thumbnail.tif',
        'bootstrapping_lower_quantile.tif',
        'bootstrapping_upper_quantile_thumbnail.tif',
        'bootstrapping_upper_quantile.tif',
        'bootstrapping_prediction_thumbnail.tif',
        'bootstrapping_prediction.tif',
        'bootstrapping_variance_thumbnail.tif',
        'bootstrapping_variance.tif',
        'bootstrapping_metadata.txt',
        'bootstrapping.model'
    ]

    @staticmethod
    @pytest.fixture(scope='class', autouse=True)
    def run_sirsam_bootstrap(request, num_procs, num_parts, sirsam_bs_conf, sirsam_bs_out):
        """
        Run the 'resample' command. Remove generated output on 
        completion.
        """

        def finalize():
            if os.path.exists(sirsam_bs_out):
                shutil.rmtree(sirsam_bs_out)

        # If running with one processor, call uncoverml directly
        if num_procs == 1:
            try:
                uncoverml.scripts.learn([sirsam_bs_conf, '-p', num_parts])
            # Catch SystemExit that gets raised by Click on competion
            except SystemExit:
                pass
            try:
                uncoverml.scripts.predict([sirsam_bs_conf, '-p', num_parts])
            except SystemExit:
                pass
        else:
            try:
                cmd = ['mpirun', '-n', str(num_procs),
                       'uncoverml', 'learn', sirsam_bs_conf, '-p', str(num_parts)]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"'{cmd}' failed with error {e.returncode}: {e.output}")
            try:
                cmd = ['mpirun', '-n', str(num_procs),
                       'uncoverml', 'predict', sirsam_bs_conf, '-p', str(num_parts)]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"'{cmd}' failed with error {e.returncode}: {e.output}")

    @classmethod
    @pytest.fixture(scope='class')
    def generated_output(cls, sirsam_bs_out):
        return [os.path.join(sirsam_bs_out, f) for f in cls.output_files]

    @staticmethod
    def test_output_exists(generated_output):
        for f in generated_output:
            assert os.path.exists(f)

    @staticmethod
    def test_model_contains_multiple_models(sirsam_bs_out):
        model, _, _ = _unpickle(os.path.join(sirsam_bs_out, 'bootstrapping.model'))
        assert hasattr(model, '__bootstrapped_model__')
        assert len(model.models) == 10
        assert all([isinstance(m, SVRTransformed) for m in model.models])


class TestResample:
    @staticmethod
    @pytest.fixture(scope='class', autouse=True)
    def run_sirsam_random_forest_resampling(request, sirsam_rs_conf, sirsam_rs_out):
        """
        Run the 'resample' command. Remove generated output on 
        completion.
        """

        def finalize():
            if os.path.exists(sirsam_rs_out):
                shutil.rmtree(sirsam_rs_out)

        request.addfinalizer(finalize)
        try:
            uncoverml.scripts.resample([sirsam_rs_conf])
        # Catch SystemExit that gets raised by Click on completion.
        except SystemExit:
            pass

    @staticmethod
    @pytest.fixture(scope='class')
    def output_filenames(sirsam_target_path):
        target_name = os.path.splitext(os.path.basename(sirsam_target_path))[0]
        exts = ['.cpg', '.dbf', '.prj', '.shp', '.shx']
        files = [target_name + '_resampled' + ext for ext in exts]
        return files

    @staticmethod
    @pytest.fixture(scope='class')
    def shapefile_output(sirsam_rs_out, output_filenames):
        return [os.path.join(sirsam_rs_out, f) for f in output_filenames]

    @staticmethod
    @pytest.fixture(scope='class')
    def precomputed_output(sirsam_rs_precomp, output_filenames):
        return [os.path.join(sirsam_rs_precomp, f) for f in output_filenames]

    @staticmethod
    def test_output_exists(shapefile_output):
        for f in shapefile_output:
            assert os.path.exists(f)

    # Outputs for resampling are random in terms of shape.
    # TODO: Work out a better comparison.
    # @staticmethod
    # def test_output_meets_baseline(shapefile_output, precomputed_output):
    #    test_shapefile = [f for f in shapefile_output if f.endswith('.shp')][0]
    #    precomp_shapefile = [f for f in precomputed_output if f.endswith('.shp')][0]

    #    test_gdf = gpd.read_file(test_shapefile)
    #    precomp_gdf = gpd.read_file(precomp_shapefile)
    #    assert abs(test_gdf.shape[0] - precomp_gdf.shape[0]) <= 20
    #    pd.testing.assert_frame_equal(test_gdf, precomp_gdf)


class TestShiftmap:
    IMAGE_OUTPUTS = [
        SIRSAM_RF + '_shiftmap_most_likely.tif',
        SIRSAM_RF + '_shiftmap_most_likely_thumbnail.tif',
        SIRSAM_RF + '_shiftmap_query_0.tif',
        SIRSAM_RF + '_shiftmap_query_0_thumbnail.tif',
        SIRSAM_RF + '_shiftmap_training_1.tif',
        SIRSAM_RF + '_shiftmap_training_1_thumbnail.tif'
    ]

    OTHER_OUTPUTS = [
        SIRSAM_RF + '_shiftmap_generated_points.csv',
    ]

    @staticmethod
    @pytest.fixture(scope='class', autouse=True)
    def run_sirsam_random_forest_shiftmap(request, num_procs, num_parts, sirsam_rf_conf,
                                          sirsam_rf_out):
        """
        Run the top level 'learn' command'. Removes generated output on
        completion.
        """

        def finalize():
            if os.path.exists(sirsam_rf_out):
                shutil.rmtree(sirsam_rf_out)

        request.addfinalizer(finalize)

        # Clear out pickled files (and other outputs)
        finalize()

        # If running with one processor, call uncoverml directly
        if num_procs == 1:
            try:
                uncoverml.scripts.shiftmap([sirsam_rf_conf, '-p', num_parts])
            # Catch SystemExit that gets raised by Click on competion
            except SystemExit:
                pass
        else:
            try:
                cmd = ['mpirun', '-n', str(num_procs),
                       'uncoverml', 'shiftmap', sirsam_rf_conf, '-p', str(num_parts)]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"'{cmd}' failed with error {e.returncode}: {e.output}")

    @staticmethod
    @pytest.fixture(params=IMAGE_OUTPUTS + OTHER_OUTPUTS)
    def sirsam_rf_output(request, sirsam_rf_out):
        return os.path.join(sirsam_rf_out, request.param)

    @staticmethod
    def test_output_exists(sirsam_rf_output):
        """
        Test that excepted outputs of 'shiftmap' command exist after
        running.
        """
        assert os.path.exists(sirsam_rf_output)

    @staticmethod
    @pytest.fixture(params=IMAGE_OUTPUTS)
    def sirsam_rf_comp_outputs(request, sirsam_rf_out, sirsam_rf_precomp_shiftmap):
        """
        """
        return (
            os.path.join(sirsam_rf_out, request.param),
            os.path.join(sirsam_rf_precomp_shiftmap, request.param)
        )

    @staticmethod
    def test_outputs_equal(sirsam_rf_comp_outputs):
        test = sirsam_rf_comp_outputs[0]
        ref = sirsam_rf_comp_outputs[1]

        with rasterio.open(test) as test_img, rasterio.open(ref) as ref_img:
            test_img_ar = test_img.read()
            ref_img_ar = ref_img.read()

        assert np.allclose(test_img_ar, ref_img_ar)


class TestLearnCommand:
    """
    Tests the 'learn' command of the UncomverML CLI.
    """
    # Group the outputs of the learn command by filetype to make them easier to test.
    SIRSAM_RF_JSON_OUTPUT = [
        SIRSAM_RF + '_crossval_scores.json',
        SIRSAM_RF + '_featureranks.json'
    ]

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
        SIRSAM_RF + '_target_scaling.png',
        SIRSAM_RF + '_real_vs_pred.png',
        SIRSAM_RF + '_residual.png'
    ]

    SIRSAM_RF_OUTPUTS = [SIRSAM_RF_FEATURE_DATA, SIRSAM_RF_TARGET_DATA,
                         SIRSAM_RF_MODEL]
    SIRSAM_RF_OUTPUTS += SIRSAM_RF_CSV_OUTPUT + SIRSAM_RF_IMAGE_OUTPUT + SIRSAM_RF_JSON_OUTPUT

    @staticmethod
    @pytest.fixture(scope='class', autouse=True)
    def run_sirsam_random_forest_learning(request, num_procs, num_parts, sirsam_rf_conf, sirsam_rf_out):
        """
        Run the top level 'learn' command'. Removes generated output on
        completion.
        """

        def finalize():
            if os.path.exists(sirsam_rf_out):
                shutil.rmtree(sirsam_rf_out)

        request.addfinalizer(finalize)

        # If running with one processor, call uncoverml directly
        if num_procs == 1:
            try:
                uncoverml.scripts.learn([sirsam_rf_conf, '-p', num_parts])
            # Catch SystemExit that gets raised by Click on competion
            except SystemExit:
                pass
        else:
            try:
                cmd = ['mpirun', '-n', str(num_procs),
                       'uncoverml', 'learn', sirsam_rf_conf, '-p', str(num_parts)]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"'{cmd}' failed with error {e.returncode}: {e.output}")

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
            os.path.join(sirsam_rf_precomp_learn, request.param)
        )

    @staticmethod
    def test_csv_outputs_match(sirsam_rf_csv_outputs):
        """
        Test that CSV covariate info matches precomputed output.
        """
        with open(sirsam_rf_csv_outputs[0]) as test, \
                open(sirsam_rf_csv_outputs[1]) as precomp:
            tl = test.readlines()
            pl = precomp.readlines()
            del tl[0]
            del pl[0]
            for t, p in zip(tl, pl):
                t_ar = []
                p_ar = []
                for x, y in zip(t.split(','), p.split(',')):
                    assert type(x) == type(y)
                    try:
                        t_ar.append(float(x))
                        p_ar.append(float(y))
                    except ValueError:
                        assert x == y
            assert np.allclose(np.array(t_ar), np.array(p_ar))

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
        for (tk, tv), (pk, pv) in zip(test_json.items(), precomp_json.items()):
            assert type(tv) == type(pv)
            if type(tv) == str:
                assert tv == pv
            elif type(tv) == dict:
                assert tv == pv
            else:
                assert np.allclose(np.array(float(tv)), np.array(float(pv)))

    @classmethod
    def test_model_outputs_match(cls, sirsam_rf_out, sirsam_rf_precomp_learn):
        """
        Test that generated model matches precomputed model.
        """
        t_model = _unpickle(os.path.join(sirsam_rf_out, cls.SIRSAM_RF_MODEL))
        p_model = _unpickle(os.path.join(sirsam_rf_precomp_learn, cls.SIRSAM_RF_MODEL))
        assert t_model == p_model

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

    @classmethod
    def test_multi_random_forest_caching(cls, sirsam_rf_out):
        model = _unpickle(os.path.join(sirsam_rf_out, cls.SIRSAM_RF_MODEL))
        assert model._randomforests == model.n_estimators


def _unpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


@pytest.mark.predict
class TestPredictCommand:
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
        # SIRSAM_RF + '_real_vs_pred.png',
        # SIRSAM_RF + '_residual.png'
    ]

    SIRSAM_RF_MF_METADATA = SIRSAM_RF + '_metadata.txt'

    @staticmethod
    @pytest.fixture(scope='class', autouse=True)
    def run_sirsam_random_forest_prediction(request, num_procs, num_parts, sirsam_rf_out, sirsam_rf_conf,
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

        # If running with one processor, call uncoverml directly
        if num_procs == 1:
            try:
                uncoverml.scripts.predict([sirsam_rf_conf, '-p', num_parts])
            # Catch SystemExit that gets raised by Click on competion
            except SystemExit:
                pass
        else:
            try:
                cmd = ['mpirun', '-n', str(num_procs),
                       'uncoverml', 'predict', sirsam_rf_conf, '-p', str(num_parts)]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"'{cmd}' failed with error {e.returncode}: {e.output}")

    @staticmethod
    @pytest.fixture(params=SIRSAM_PREDICTION_MAPS + [SIRSAM_RF_MF_METADATA] + SIRSAM_RF_IMAGE_OUTPUT)
    def sirsam_rf_output(request, sirsam_rf_out):
        return os.path.join(sirsam_rf_out, request.param)

    @staticmethod
    @pytest.fixture(params=SIRSAM_PREDICTION_MAPS)
    def sirsam_rf_comp_outputs(request, sirsam_rf_out, sirsam_rf_precomp_predict):
        """
        """
        return (
            os.path.join(sirsam_rf_out, request.param),
            os.path.join(sirsam_rf_precomp_predict, request.param)
        )

    @staticmethod
    def test_output_exists(sirsam_rf_output):
        """
        Test that excepted outputs of 'predict' command exist after
        running.
        """
        assert os.path.exists(sirsam_rf_output)

    @staticmethod
    def test_outputs_equal(sirsam_rf_comp_outputs):
        test = sirsam_rf_comp_outputs[0]
        ref = sirsam_rf_comp_outputs[1]

        with rasterio.open(test) as test_img, rasterio.open(ref) as ref_img:
            test_img_ar = test_img.read()
            ref_img_ar = ref_img.read()

        assert np.allclose(test_img_ar, ref_img_ar)
