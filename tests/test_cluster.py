import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from numpy.testing import assert_allclose

from uncoverml.cluster import (
    KMeans, kmeans_step, initialise_centres,
    compute_n_classes, extract_data, calc_cluster_dist, run_kmeans
)

@pytest.fixture
def simple_data():
    np.random.seed(0)
    cluster1 = np.random.randn(10, 2) + np.array([0, 0])
    cluster2 = np.random.randn(10, 2) + np.array([5, 5])
    return np.vstack([cluster1, cluster2])


def test_reseed_point_returns_valid_point(simple_data):
    kmeans = KMeans(2, mpiops=None)
    point = kmeans._reseed_point(simple_data)
    assert point.shape == (2,)
    assert np.any(np.all(point == simple_data, axis=1))


@patch('uncoverml.cluster.mpiops')
def test_initialise_seeds_with_mocks(mpiops_mock, simple_data):
    mpiops_mock.comm.rank = 0
    mpiops_mock.scatter.return_value = simple_data
    mpiops_mock.broadcast.side_effect = lambda x: x
    mpiops_mock.gather.return_value = [simple_data]

    kmeans = KMeans(2, mpiops=mpiops_mock)
    initial_centroids = kmeans._initialise(simple_data)

    assert initial_centroids.shape == (2, 2)
    assert np.all(np.isfinite(initial_centroids))


@patch('uncoverml.cluster.mpiops')
def test_kmeans_learn_basic_clustering(mpiops_mock, simple_data):
    mpiops_mock.rank = 0
    mpiops_mock.comm.rank = 0
    mpiops_mock.scatter.return_value = simple_data
    mpiops_mock.broadcast.side_effect = lambda x: x
    mpiops_mock.gather.return_value = [simple_data]
    mpiops_mock.reduce.return_value = len(simple_data)

    kmeans = KMeans(2, mpiops=mpiops_mock)
    centroids, responsibilities = kmeans.learn(simple_data)

    assert centroids.shape == (2, 2)
    assert responsibilities.shape == (20,)
    assert np.all(np.isin(responsibilities, [0, 1]))


@patch('uncoverml.cluster.mpiops')
def test_kmeans_step_basic(mpiops_mock):
    X = np.array([[0], [10], [20]])
    C = np.array([[0], [20]])
    classes = np.array([0, 0, 1])

    mpiops_mock.comm.allreduce.side_effect = lambda x, op=None: x

    result = kmeans_step(X, C, classes)
    expected = np.array([[5], [20]])
    assert_allclose(result, expected)


@patch('uncoverml.cluster.mpiops')
def test_run_kmeans_converges(mpiops_mock):
    X = np.array([[0], [1], [10], [11]])
    C = np.array([[0], [10]])
    k = 2

    mpiops_mock.comm.allreduce.side_effect = lambda x, op=None: x

    C_final, classes = run_kmeans(X, C, k)
    assert C_final.shape == (2, 1)
    assert set(classes) == {0, 1}


@patch("uncoverml.cluster.weighted_starting_candidates")
@patch("uncoverml.cluster.run_kmeans")
@patch("uncoverml.cluster.mpiops.comm.bcast")
def test_initialise_centres(mock_bcast, mock_run_kmeans, mock_weighted_starting):
    X = np.random.rand(10, 2)
    C = np.random.rand(20, 2)
    mock_weighted_starting.return_value = (np.ones(20), C)
    mock_bcast.return_value = np.arange(3)
    mock_run_kmeans.return_value = (np.random.rand(3, 2), None)

    result = initialise_centres(X, 3, 2.0)
    assert result.shape == (3, 2)


@patch("uncoverml.cluster.mpiops.comm.allreduce", return_value=3)
@patch("uncoverml.cluster.mpiops.MPI")
def test_compute_n_classes_respects_training(mock_mpi, mock_allreduce):
    class Config:
        n_classes = 3

    mock_mpi.MAX = None  # placeholder, not used
    classes = np.array([0, 1, 2, 3])
    result = compute_n_classes(classes, Config())
    assert result == 3


@patch("uncoverml.cluster.rasterio.open")
def test_extract_data(mock_open):
    class DummySrc:
        def sample(self, coords):
            return [[1.0] for _ in coords]
        def __enter__(self): return self
        def __exit__(self, *args): pass

    mock_open.return_value = DummySrc()
    result = extract_data("dummy.tif", [(0, 0), (1, 1)])
    assert np.all(result == np.array([1.0, 1.0]))


def test_calc_cluster_dist_identity():
    centres = np.array([[0.0, 0.0], [3.0, 4.0]])
    dist_mat = calc_cluster_dist(centres)
    assert_allclose(dist_mat[0, 1], 5.0)
    assert_allclose(dist_mat[1, 0], 5.0)
