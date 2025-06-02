import numpy as np
import pytest
from numpy.testing import assert_allclose
from unittest.mock import MagicMock, patch

from unittest import mock

from uncoverml.cluster import (
    KMeans,
    kmeans_step,
    initialise_centres,
    compute_n_classes,
    extract_data,
    calc_cluster_dist,
    run_kmeans
)

@pytest.fixture
def simple_data():
    np.random.seed(0)
    c1 = np.random.randn(10, 2)
    c2 = np.random.randn(10, 2) + 5
    return np.vstack([c1, c2])

def test_kmeans_learn_basic_clustering(simple_data):
    km = KMeans(2, 2.0)
    centres, labels = km.learn(simple_data)
    assert centres.shape == (2, 2)
    assert labels.shape == (20,)
    assert np.all(np.isin(labels, [0, 1]))

def test_kmeans_step_basic():
    X = np.array([[0], [10], [20]])
    C = np.array([[0], [20]])
    classes = np.array([0, 0, 1])
    dummy_mpi = MagicMock()
    dummy_mpi.allreduce.side_effect = lambda x, op=None: x

    with patch("uncoverml.cluster.mpiops.comm", dummy_mpi):
        result = kmeans_step(X, C, classes)
        assert_allclose(result, np.array([[5.0], [20.0]]))

def test_run_kmeans_converges():
    X = np.array([[0], [1], [10], [11]])
    C = np.array([[0], [10]])
    dummy_mpi = MagicMock()
    dummy_mpi.allreduce.side_effect = lambda x, op=None: x

    with patch("uncoverml.cluster.mpiops.comm", dummy_mpi):
        final_C, classes = run_kmeans(X, C, max_iterations=2)
        assert final_C.shape == (2, 1)
        assert set(classes) == {0, 1}

@patch("uncoverml.cluster.rasterio.open")
def test_extract_data(mock_open):
    class DummySrc:
        def sample(self, coords): return [[1.0] for _ in coords]
        def __enter__(self): return self
        def __exit__(self, *args): pass

    mock_open.return_value = DummySrc()
    out = extract_data("dummy.tif", [(0, 0), (1, 1)])
    assert np.all(out == np.array([1.0, 1.0]))

def test_calc_cluster_dist_identity():
    C = np.array([[0, 0], [3, 4]])
    dist = calc_cluster_dist(C)
    assert_allclose(dist[0, 1], 5.0)
    assert_allclose(dist[1, 0], 5.0)

def test_compute_n_classes_respects_training():
    class Cfg: n_classes = 3
    cls = np.array([0, 1, 2, 2])
    dummy_mpi = MagicMock()
    dummy_mpi.comm.allreduce.return_value = 3
    dummy_mpi.MPI.MAX = None

    with patch("uncoverml.cluster.mpiops", dummy_mpi):
        assert compute_n_classes(cls, Cfg()) == 3

