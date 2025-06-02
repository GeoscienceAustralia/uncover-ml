import numpy as np
import pytest
from unittest import mock

from uncoverml.cluster.kmeans import KMeans
from uncoverml.cluster import kmeans_step, \
    initialise_centres, \
        compute_n_classes, \
            extract_data, calc_cluster_dist

from numpy.testing import assert_allclose
from unittest.mock import patch


@pytest.fixture
def simple_data():
    # Small synthetic dataset: 2 clusters in 2D
    np.random.seed(0)
    cluster1 = np.random.randn(10, 2) + np.array([0, 0])
    cluster2 = np.random.randn(10, 2) + np.array([5, 5])
    return np.vstack([cluster1, cluster2])


def test_reseed_point_returns_valid_point(simple_data):
    kmeans = KMeans(n_clusters=2, mpiops=None)
    point = kmeans._reseed_point(simple_data)
    assert point.shape == (2,)
    assert np.any(np.all(point == simple_data, axis=1))  # point must be from input


@mock.patch('uncoverml.cluster.kmeans.mpiops')
def test_initialise_seeds_with_mocks(mpiops_mock, simple_data):
    # Make MPI rank 0 return full dataset, others get empty
    mpiops_mock.comm.rank = 0
    mpiops_mock.scatter.return_value = simple_data
    mpiops_mock.broadcast.side_effect = lambda x: x
    mpiops_mock.gather.return_value = [simple_data]

    kmeans = KMeans(n_clusters=2, mpiops=mpiops_mock)
    initial_centroids = kmeans._initialise(simple_data)

    assert initial_centroids.shape == (2, 2)
    assert np.all(np.isfinite(initial_centroids))


@mock.patch('uncoverml.cluster.kmeans.mpiops')
def test_kmeans_learn_basic_clustering(mpiops_mock, simple_data):
    # Mock MPI behavior: single-process test
    mpiops_mock.rank = 0
    mpiops_mock.comm.rank = 0
    mpiops_mock.scatter.return_value = simple_data
    mpiops_mock.broadcast.side_effect = lambda x: x
    mpiops_mock.gather.return_value = [simple_data]
    mpiops_mock.reduce.return_value = len(simple_data)

    kmeans = KMeans(n_clusters=2, mpiops=mpiops_mock)
    centroids, responsibilities = kmeans.learn(simple_data)

    assert centroids.shape == (2, 2)
    assert responsibilities.shape == (20,)
    assert np.all((responsibilities == 0) | (responsibilities == 1))


def test_kmeans_step_basic():
    X = np.array([[0], [10], [20]])
    C = np.array([[0], [20]])
    classes = np.array([0, 0, 1])
    expected = np.array([[5], [20]])

    with patch("your_module.mpiops.comm.allreduce", side_effect=lambda x, op=None: x):
        result = kmeans_step(X, C, classes)
        assert_allclose(result, expected)


def test_run_kmeans_converges():
    X = np.array([[0], [1], [10], [11]])
    C = np.array([[0], [10]])
    k = 2

    with patch("your_module.mpiops.comm.allreduce", side_effect=lambda x, op=None: x):
        C_final, classes = run_kmeans(X, C, k)
        assert C_final.shape == (2, 1)
        assert set(classes) == {0, 1}


@patch("your_module.weighted_starting_candidates")
@patch("your_module.run_kmeans")
@patch("your_module.mpiops.comm.bcast")
def test_initialise_centres(mock_bcast, mock_run_kmeans, mock_weighted_starting):
    X = np.random.rand(10, 2)
    C = np.random.rand(20, 2)
    mock_weighted_starting.return_value = (np.ones(20), C)
    mock_bcast.return_value = np.arange(3)
    mock_run_kmeans.return_value = (np.random.rand(3, 2), None)

    result = initialise_centres(X, 3, 2.0)
    assert result.shape == (3, 2)


def test_compute_n_classes_respects_training():
    class Config:
        n_classes = 3

    classes = np.array([0, 1, 2, 3])
    with patch("your_module.mpiops.comm.allreduce", return_value=3):
        with patch("your_module.mpiops.MPI.MAX", new=None):
            result = compute_n_classes(classes, Config())
            assert result == 3


@patch("your_module.rasterio.open")
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
    centres = np.array([[0, 0], [3, 4]])
    dist_mat = calc_cluster_dist(centres)
    assert_allclose(dist_mat[0, 1], 5.0)
    assert_allclose(dist_mat[1, 0], 5.0)
