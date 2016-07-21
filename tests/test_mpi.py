import copy

import numpy as np
import pytest

from uncoverml import mpiops
from uncoverml import pipeline


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


def test_run_if(mpisync):
    # This test only works on multiple nodes
    if mpiops.chunks == 1:
        assert True
        return
    idx = mpiops.chunk_index

    def f(x, comm):
        return x + comm.rank  # this is rank in the split comm

    flag = idx != 0
    result = mpiops.run_if(f, flag, x=0)
    true_result = None if idx == 0 else idx - 1
    assert result == true_result


def test_run_if_broadcast(mpisync):
    # This test only works on multiple nodes
    if mpiops.chunks == 1:
        assert True
        return
    idx = mpiops.chunk_index

    def f(x, comm):
        return x + comm.rank  # this is rank in the split comm

    flag = idx != 0
    result = mpiops.run_if(f, flag, x=0, broadcast=True)
    true_result = 0
    assert result == true_result


@pytest.fixture(params=['none', 'centre', 'standardise', 'whiten'])
def transform_opt(request):
    return request.param


@pytest.fixture(params=[True, False])
def impute_opt(request):
    return request.param


@pytest.fixture(params=[0.2, 0.8, 1.0])
def feature_opt(request):
    return request.param


def test_compose_transform_none(mpisync, impute_opt, feature_opt):
    x_data = np.ones((10, 2))
    x_data[:, 1] = 2.0

    x_mask = np.zeros_like(x_data, dtype=bool)
    x_mask[5:] = True
    x_data[5:] = 0.0

    x = np.ma.array(data=x_data, mask=x_mask)
    x_in = np.ma.copy(x)

    settings = pipeline.ComposeSettings(impute=impute_opt,
                                        transform='none',
                                        featurefraction=feature_opt,
                                        impute_mean=None,
                                        mean=None, sd=None, eigvals=None,
                                        eigvecs=None)
    settings_in = copy.deepcopy(settings)
    x_r, settings_r = mpiops._compose_transform(x_in, settings_in, mpiops.comm)
    assert np.all(x == x_r)
    if impute_opt:
        mask_true = np.zeros_like(x_data, dtype=bool)
        impute_mean_true = x_data[0]
        assert np.all(impute_mean_true == settings_r.impute_mean)
    else:
        mask_true = x_mask

    assert np.all(mask_true == x_r.mask)


def test_compose_transform_centre(mpisync, feature_opt, impute_opt):
    x_data = np.ones((10, 2))
    x_data[:, 1] = 2.0

    x_mask = np.zeros_like(x_data, dtype=bool)
    x_mask[5:] = True
    x_data[5:] = 0.0

    x = np.ma.array(data=x_data, mask=x_mask)
    x_in = np.ma.copy(x)

    settings = pipeline.ComposeSettings(impute=impute_opt,
                                        transform='centre',
                                        featurefraction=feature_opt,
                                        impute_mean=None,
                                        mean=None, sd=None, eigvals=None,
                                        eigvecs=None)
    settings_in = copy.deepcopy(settings)
    x_r, settings_r = mpiops._compose_transform(x_in, settings_in, mpiops.comm)


    assert np.all(x == x_r)
    if impute_opt:
        mask_true = np.zeros_like(x_data, dtype=bool)
        impute_mean_true = x_data[0]
        assert np.all(impute_mean_true == settings_r.impute_mean)
    else:
        mask_true = x_mask



