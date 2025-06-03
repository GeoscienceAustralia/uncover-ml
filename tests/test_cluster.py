import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import tempfile
import os
import joblib
from unittest import mock
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from types import SimpleNamespace
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from os import path
from uncoverml.cluster import (
    KMeans, kmeans_step, run_kmeans, calc_cluster_dist,
    split_all_feat_data, split_save_feat_clusters, split_pred_parallel,
    process_and_save_data, training_data_boxplot
)


def test_kmeans_learn_basic_clustering():
    x = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(k=2, oversample_factor=2)
    kmeans.learn(x)
    assert kmeans.centres.shape == (2, 2)


def test_kmeans_step_basic():
    x = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    initial_centres = np.array([[1, 2], [10, 2]])
    dists = np.linalg.norm(x[:, np.newaxis, :] - initial_centres[np.newaxis, :, :], axis=2)
    classes = np.argmin(dists, axis=1)
    new_centres = kmeans_step(x, initial_centres, classes)
    assert new_centres.shape == (2, 2)


def test_run_kmeans_converges():
    x = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    initial_centres = np.array([[1, 2], [10, 2]])
    final_centres, labels = run_kmeans(x, initial_centres, k=2, max_iterations=100)
    assert final_centres.shape == (2, 2)
    assert labels.shape == (6,)


def test_calc_cluster_dist_identity():
    centres = np.array([[1, 2], [10, 2]])
    dist = calc_cluster_dist(centres)
    assert dist.shape == (2, 2)
    assert np.all(dist >= 0)
    assert np.allclose(np.diag(dist), 0)


def test_calc_cluster_dist_identity():
    x = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    centres = np.array([[1, 2], [10, 2]])
    dist = np.linalg.norm(x[:, np.newaxis] - centres, axis=2)
    assert dist.shape == (6, 2)
    assert np.all(dist >= 0)


def test_process_and_save_data():
    feat_data = np.array([[10, 20], [30, 40]])
    pred_data = np.array([[0, 1], [0, 1]])
    clust_num = 0
    no_data_val = -9999

    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tmpfile:
        process_and_save_data(feat_data, pred_data, tmpfile, clust_num, no_data_val)
        tmpfile.seek(0)
        lines = tmpfile.readlines()

    os.remove(tmpfile.name)
    assert len(lines) == 2


def test_split_save_feat_clusters():
    # Dummy data
    feat_data = np.array([[10, 20], [30, 40]], dtype=np.int32)
    pred_data = np.array([[0, 1], [1, 0]], dtype=np.int32)
    n_classes = 2
    feat_name = "testfeat"
    no_data_val = -9999

    with tempfile.TemporaryDirectory() as tmpdir:
        feat_path = os.path.join(tmpdir, "features.tif")
        pred_path = os.path.join(tmpdir, "predictions.tif")

        transform = from_origin(0, 2, 1, 1)

        with rasterio.open(
            feat_path, 'w',
            driver='GTiff',
            height=feat_data.shape[0],
            width=feat_data.shape[1],
            count=1,
            dtype=feat_data.dtype,
            nodata=no_data_val,
            transform=transform
        ) as dst:
            dst.write(feat_data, 1)

        with rasterio.open(
            pred_path, 'w',
            driver='GTiff',
            height=pred_data.shape[0],
            width=pred_data.shape[1],
            count=1,
            dtype=pred_data.dtype,
            nodata=no_data_val,
            transform=transform
        ) as dst:
            dst.write(pred_data, 1)

        main_config = SimpleNamespace(output_dir=tmpdir)

        with rasterio.open(feat_path) as feat_ds, rasterio.open(pred_path) as pred_ds:
            split_save_feat_clusters(main_config, feat_ds, pred_ds, feat_name, n_classes)

        for i in range(n_classes):
            out_csv = os.path.join(tmpdir, f"feat_{feat_name}_clust_{i}.csv")
            assert os.path.exists(out_csv), f"Missing output: {out_csv}"
            with open(out_csv) as f:
                lines = f.readlines()
                assert len(lines) > 0, f"Output file {out_csv} is empty"


def test_split_all_feat_data():
    pred = np.array([[0, 1], [1, 0]])
    feat = np.array([[10, 20], [30, 40]])

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_path = os.path.join(tmpdir, 'kmeans_class.tif')
        feat_path = os.path.join(tmpdir, 'feat.tif')

        def create_raster(path, data):
            with rasterio.open(
                path, 'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype='int32',
                nodata=-9999
            ) as dst:
                dst.write(data, 1)

        create_raster(pred_path, pred)
        create_raster(feat_path, feat)

        mock_config = mock.Mock()
        mock_config.output_dir = tmpdir
        mock_config.n_classes = 2
        mock_config.feature_sets = [mock.Mock(files=[feat_path])]
        mock_config.short_names = ['test']

        split_all_feat_data(mock_config)

        assert os.path.exists(os.path.join(tmpdir, 'feat_test_clust_0.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'feat_test_clust_1.csv'))


def test_split_pred_parallel_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_data = np.array([[0, 1], [1, 0]], dtype=np.int32)
        feat_data = np.array([[10, 20], [30, 40]], dtype=np.int32)

        def write_raster(filepath, data):
            with rasterio.open(
                filepath, 'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype='int32',
                nodata=-9999
            ) as dst:
                dst.write(data, 1)

        pred_path = path.join(tmpdir, 'kmeans_class.tif')
        feat_path = path.join(tmpdir, 'feat.tif')
        write_raster(pred_path, pred_data)
        write_raster(feat_path, feat_data)

        config = mock.Mock()
        config.output_dir = tmpdir
        config.n_classes = 2
        config.feature_sets = [mock.Mock(files=[feat_path])]
        config.short_names = ['test']

        split_pred_parallel(config)

        assert os.path.exists(path.join(tmpdir, 'feat_test_clust_0.csv'))
        assert os.path.exists(path.join(tmpdir, 'feat_test_clust_1.csv'))


def test_process_and_save_data_cluster_match():
    pred = np.array([[0, 1], [1, 0]])
    feat = np.array([[10, 20], [30, 40]])
    out = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    try:
        process_and_save_data(feat, pred, out, clust_num=1, no_data_val=-9999)
        out.seek(0)
        contents = out.readlines()
        assert len(contents) > 0
    finally:
        out.close()
        os.remove(out.name)


class DummyModel:
    def predict(self, data):
        return np.array([0, 1, 0, 1])

class DummyFeatureSet:
    def __init__(self, files):
        self.files = files

class DummyConfig:
    def __init__(self, output_dir):
        self.n_classes = 2
        self.output_dir = output_dir
        self.feature_sets = [DummyFeatureSet(files=['dummy1', 'dummy2'])]
        self.short_names = ['f1', 'f2']

def test_training_data_boxplot_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = DummyModel()
        config = DummyConfig(tmpdir)
        training_data = np.array([
            [1, 10],
            [2, 20],
            [1.5, 12],
            [2.5, 22]
        ])

        model_path = path.join(tmpdir, 'model.joblib')
        training_path = path.join(tmpdir, 'training.joblib')

        joblib.dump({'model': model, 'config': config}, model_path)
        joblib.dump(training_data, training_path)

        training_data_boxplot(model_path, training_path)

        assert os.path.exists(path.join(tmpdir, 'training_data_boxplots.png'))
