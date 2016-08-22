import copy

import numpy as np
import pytest

from uncoverml import mpiops
from uncoverml import pipeline
from uncoverml import transforms


# Make sure all MPI tests use this fixure
@pytest.fixture()
def mpisync(request):
    mpiops.comm.barrier()

    def fin():
        mpiops.comm.barrier()
    request.addfinalizer(fin)
    return mpiops.comm


def test_helloworld(mpisync):
    comm = mpiops.comm
    ranks = comm.allgather(mpiops.chunk_index)
    assert len(ranks) == mpiops.chunks


@pytest.fixture(params=['none', 'centre', 'standardise', 'whiten'])
def transform_opt(request):
    return request.param


@pytest.fixture(params=[True, False])
def impute_opt(request):
    return request.param


@pytest.fixture(params=[0.2, 0.8, 1.0])
def feature_opt(request):
    return request.param


@pytest.fixture
def masked_array():
    np.random.seed(mpiops.chunk_index)
    x_data = np.random.rand(10, 2) * 5 + 10
    x_mask = np.random.choice([False, False, True], size=(10, 2))
    x = np.ma.array(data=x_data, mask=x_mask)
    x_all = np.ma.concatenate(mpiops.comm.allgather(x), axis=0)
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


class DummySettings:
    def __init__(self):
        pass


def test_sd(mpisync, masked_array):
    x, x_all = masked_array
    sd = mpiops.sd(x)
    sd_true = np.ma.std(x_all, axis=0, ddof=0).data
    assert np.allclose(sd, sd_true)


# def test_impute(mpisync, masked_array):
#     x, x_all = masked_array
#     settings = DummySettings()
#     settings.impute_mean = None
#     x_im = mpiops._impute(x, settings, mpiops.comm)

#     x_mean = np.ma.mean(x_all, axis=0).data
#     x_im_true = np.ma.copy(x)
#     transforms.impute_with_mean(x_im_true, x_mean)

#     assert np.allclose(x_im_true, x_im)
#     assert np.allclose(x_mean, settings.impute_mean)
#     print(settings.impute_mean)


# def test_impute_params(mpisync, masked_array):
#     x, x_all = masked_array
#     settings = DummySettings()
#     settings.impute_mean = np.ones(x.shape[1])
#     x_im = mpiops._impute(x, settings, mpiops.comm)
#     x_im_true = np.ma.copy(x)
#     transforms.impute_with_mean(x_im_true, np.ones(x.shape[1]))
#     assert np.allclose(x_im_true, x_im)
#     assert np.all(settings.impute_mean == np.ones(x.shape[1]))



# def test_centre(mpisync, masked_array):
#     x, x_all = masked_array
#     settings = DummySettings()
#     settings.mean = None
#     x_centre = copy.deepcopy(x)
#     x_centre = mpiops.centre(x_centre, settings, mpiops.comm)
#     true_mean = np.ma.mean(x_all, axis=0)
#     x_true = x - true_mean
#     assert(np.allclose(x_true, x_centre))
#     assert(np.allclose(true_mean, settings.mean))


# def test_centre_param(mpisync, masked_array):
#     x, x_all = masked_array
#     settings = DummySettings()
#     settings.mean = np.ones(x.shape[1])
#     x_centre = copy.deepcopy(x)
#     x_centre = mpiops.centre(x_centre, settings, mpiops.comm)
#     x_true = x - np.ones(x.shape[1])
#     assert(np.allclose(x_true, x_centre))


# def test_standardise(mpisync, masked_array):
#     x, x_all = masked_array
#     settings = DummySettings()
#     settings.sd = None
#     settings.mean = None
#     x_stand = copy.deepcopy(x)
#     x_stand = mpiops.standardise(x_stand, settings, mpiops.comm)
#     x_demean = (x - np.ma.mean(x_all, axis=0))
#     sd_true = np.ma.std(x_all, axis=0)
#     x_true = x_demean / sd_true
#     assert np.allclose(x_true, x_stand)
#     assert np.allclose(sd_true, settings.sd)


# def test_standardise_param(mpisync, masked_array):
#     x, x_all = masked_array
#     settings = DummySettings()
#     settings.sd = np.ones(x.shape[1]) * 2
#     settings.mean = np.ones(x.shape[1])
#     x_stand = copy.deepcopy(x)
#     x_stand = mpiops.standardise(x_stand, settings, mpiops.comm)
#     x_demean = (x - 1)
#     x_true = x_demean / 2.0
#     assert np.allclose(x_true, x_stand)


# def test_whiten(mpisync, masked_array):
#     x, x_all = masked_array
#     settings = DummySettings()
#     settings.mean = None
#     settings.sd = None
#     settings.eigvals = None
#     settings.eigvecs = None
#     settings.featurefraction = 0.5
#     x_white = copy.deepcopy(x)
#     x_white = mpiops.whiten(x_white, settings, mpiops.comm)

#     x_true_mean = np.ma.mean(x_all, axis=0)
#     x_demean = x_all - x_true_mean
#     true_cov = np.ma.dot(x_demean.T, x_demean)/np.ma.count(x_demean, axis=0)
#     eigvals, eigvecs = np.linalg.eigh(true_cov)

#     keepdims = 1
#     mat = eigvecs[:, -keepdims:]
#     vec = eigvals[-keepdims:]
#     x_true = np.ma.dot(x - x_true_mean, mat, strict=True) / np.sqrt(vec)
#     assert np.allclose(x_white, x_true)
#     assert np.allclose(eigvals, settings.eigvals)
#     assert np.allclose(eigvecs, settings.eigvecs)


# def test_whiten_params(mpisync, masked_array):
#     x, x_all = masked_array
#     settings = DummySettings()
#     settings.mean = np.ones(2)
#     settings.sd = np.ones(2)*2
#     settings.eigvals = np.ones(2)*3
#     settings.eigvecs = np.eye(2)*4
#     settings.featurefraction = 0.5
#     x_white = copy.deepcopy(x)
#     x_white = mpiops.whiten(x_white, settings, mpiops.comm)

#     x_true_mean = settings.mean

#     keepdims = 1
#     mat = settings.eigvecs[:, -keepdims:]
#     vec = settings.eigvals[-keepdims:]
#     x_true = np.ma.dot(x - x_true_mean, mat, strict=True) / np.sqrt(vec)
#     assert np.allclose(x_white, x_true)

