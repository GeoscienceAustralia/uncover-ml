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
    no_data_val = -9999  # or np.nan

    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tmpfile:
        process_and_save_data(feat_data, pred_data, tmpfile, clust_num, no_data_val)
        tmpfile.seek(0)
        lines = tmpfile.readlines()

    os.remove(tmpfile.name)
    assert len(lines) == 2  # Two values for cluster 0


def test_split_save_feat_clusters():
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_data = np.array([[0, 1], [1, 0]])
        feat_data = np.array([[10, 20], [30, 40]])

        class DummySrc:
            def __init__(self, data):
                self.data = data
                self.width = data.shape[1]
                self.height = data.shape[0]
                self.nodata = -9999
            def read(self, band, window=None):
                return self.data[window.row_off:window.row_off+window.height,
                                 window.col_off:window.col_off+window.width]

        config = mock.Mock()
        config.output_dir = tmpdir

        split_save_feat_clusters(config, DummySrc(feat_data), DummySrc(pred_data), 'testfeat', 2)

        out0 = np.loadtxt(f"{tmpdir}/feat_testfeat_clust_0.csv")
        out1 = np.loadtxt(f"{tmpdir}/feat_testfeat_clust_1.csv")

        assert out0.size > 0
        assert out1.size > 0


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


def test_training_data_boxplot_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = mock.Mock()
        model.predict.return_value = np.array([0, 1, 0, 1])
        config = mock.Mock()
        config.n_classes = 2
        config.output_dir = tmpdir
        config.feature_sets = [mock.Mock(files=['dummy1', 'dummy2'])]
        config.short_names = ['f1', 'f2']
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
