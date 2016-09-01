from scipy.stats import bernoulli

from uncoverml.transforms.onehot import sets, one_hot
from uncoverml.transforms.impute import impute_with_mean, GaussImputer, \
    NearestNeighboursImputer
from uncoverml.transforms import target
import numpy as np

import pytest


@pytest.fixture(params=list(target.transforms.keys()))
def get_transform_names(request):
    return request.param


def test_transform(get_transform_names):

    y = np.concatenate((np.random.randn(100), np.random.randn(100) + 5))
    transformer = target.transforms[get_transform_names]()

    if hasattr(transformer, 'offset'):
        y -= (y.min() - 1e-5)

    transformer.fit(y)
    yt = transformer.transform(y)
    yr = transformer.itransform(yt)

    assert np.allclose(yr, y)


def test_sets(int_masked_array):
    r = sets(int_masked_array)
    assert(np.all(r == np.array([[2, 3, 4], [1, 3, 5]], dtype=int)))


@pytest.fixture
def make_missing_data():

    N = 100

    Xdata = np.random.randn(N, 2)
    Xmask = bernoulli.rvs(p=0.3, size=(N, 2)).astype(bool)
    Xmask[Xmask[:, 0], 1] = False  # Make sure we don't have all missing rows

    X = np.ma.array(data=Xdata, mask=Xmask)

    return X


def test_GaussImputer(make_missing_data):

    X = make_missing_data
    Xcopy = X.copy()
    imputer = GaussImputer()
    Ximp = imputer(X)

    assert np.all(~np.isnan(Ximp))
    assert np.all(~np.isinf(Ximp))
    assert Ximp.shape == Xcopy.shape
    assert np.ma.count_masked(Ximp) == 0


def test_NearestNeighbourImputer(make_missing_data):

    X = make_missing_data
    Xcopy = X.copy()
    nn = np.ma.MaskedArray(data=np.random.randn(100, 2), mask=False)

    imputer = NearestNeighboursImputer(nodes=100)
    imputer(nn)

    Ximp = imputer(X)

    assert np.all(~np.isnan(Ximp))
    assert np.all(~np.isinf(Ximp))
    assert Ximp.shape == Xcopy.shape
    assert np.ma.count_masked(Ximp) == 0

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


# def test_impute_with_mean(masked_array):
#     t = np.ma.copy(masked_array)
#     mean = np.ma.mean(t, axis=0)
#     trueval = np.array([[3.0,  1.0], [2.0, 3.0], [4.0, 5.0]])
#     impute_with_mean(t, mean)
#     assert np.all(t.data == trueval)
#     assert np.all(np.logical_not(t.mask))


# def test_one_hot(int_masked_array):
#     x_set = np.array([[2, 3, 4], [1, 3, 5]], dtype=int)
#     r = one_hot(int_masked_array,  x_set)
#     ans = np.array([[-0.5,  -0.5,  -0.5,   0.5,  -0.5,  -0.5],
#                    [0.5,  -0.5,  -0.5,  -0.5,  -0.5,  -0.5],
#                    [-0.5,  -0.5,   0.5,  -0.5,  -0.5,   0.5],
#                    [-0.5,   0.5,  -0.5,  -0.5,   0.5,  -0.5]])
#     ans_mask = np.array(
#         [[True,   True,   True,  False,  False,  False],
#          [False,  False,  False,   True,   True,   True],
#          [False,  False,  False,  False,  False,  False],
#          [False,  False,  False,  False,  False,  False]],  dtype=bool)
#     assert np.all(r.data == ans)
#     assert np.all(r.mask == ans_mask)
