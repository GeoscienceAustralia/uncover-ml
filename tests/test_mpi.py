# import copy

import numpy as np
import pytest

from uncoverml import mpiops
# from uncoverml import pipeline
# from uncoverml import transforms


def test_helloworld(mpisync):
    comm = mpiops.comm_world
    ranks = comm.allgather(mpiops.rank_world)
    assert len(ranks) == mpiops.size_world


@pytest.fixture(params=['none', 'centre', 'standardise', 'whiten'])
def transform_opt(request):
    return request.param


def test_power(masked_array):
    x, _ = masked_array
    x2 = mpiops.power(x, 2)
    assert np.allclose(x2, x**2)


@pytest.fixture(params=[True, False])
def impute_opt(request):
    return request.param


@pytest.fixture(params=[0.2, 0.8, 1.0])
def feature_opt(request):
    return request.param


@pytest.fixture
def masked_array():
    np.random.seed(mpiops.rank_world)
    x_data = np.random.rand(10, 2) * 5 + 10
    x_mask = np.random.choice([False, False, True], size=(10, 2))
    x = np.ma.array(data=x_data, mask=x_mask)
    x_all = np.ma.concatenate(mpiops.comm_world.allgather(x), axis=0)
    return x, x_all


def test_count(mpisync, masked_array):
    x, x_all = masked_array
    x_n = mpiops.count(x)
    x_n_true = x_all.count(axis=0)
    assert np.all(x_n == x_n_true)


def test_mean(mpisync, masked_array):
    x, x_all = masked_array
    m = mpiops.mean(x)
    m_true = np.ma.mean(x_all, axis=0).data
    assert np.allclose(m_true, m)


def test_covariance(mpisync, masked_array):
    x, x_all = masked_array
    c = mpiops.covariance(x)
    c_true = np.ma.cov(x_all.T, bias=True).data
    assert np.allclose(c_true, c)


class DummySettings:
    def __init__(self):
        pass


def test_sd(mpisync, masked_array):
    x, x_all = masked_array
    sd = mpiops.sd(x)
    sd_true = np.ma.std(x_all, axis=0, ddof=0).data
    assert np.allclose(sd, sd_true)


def test_random_full_points():

    Xd = np.random.randn(100, 3)
    Xm = np.zeros_like(Xd, dtype=bool)
    Xm[30, 1] = True
    Xm[67, 2] = True
    X = np.ma.MaskedArray(data=Xd, mask=Xm)

    Xp = mpiops.random_full_points(X, 80)

    assert Xp.shape[1] == 3
    assert np.ma.count_masked(Xp) == 0

    Xp = mpiops.random_full_points(X, 200)

    assert Xp.shape[0] <= 100
