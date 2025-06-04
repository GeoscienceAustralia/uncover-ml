from scipy.stats import bernoulli

from uncoverml.transforms.onehot import sets
from uncoverml.transforms.impute import (GaussImputer,
                                         NearestNeighboursImputer, MeanImputer)
from uncoverml.transforms.linear import (CentreTransform, StandardiseTransform,
                                         WhitenTransform, LogTransform,
                                         SqrtTransform)
from uncoverml.transforms import target
import numpy as np

import pytest

SEED = 1


@pytest.fixture(params=list(target.transforms.keys()))
def get_transform_names(request):
    return request.param


def test_transform(get_transform_names):

    rnd = np.random.RandomState(SEED)
    y = np.concatenate((rnd.randn(100), rnd.randn(100) + 5))
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

    rnd = np.random.RandomState(SEED)
    Xdata = rnd.randn(N, 2)
    Xmask = bernoulli.rvs(p=0.3, size=(N, 2)).astype(bool)
    Xmask[Xmask[:, 0], 1] = False

    X = np.ma.array(data=Xdata, mask=Xmask)

    return X


@pytest.fixture()
def make_random_data(n=10, m=3):

    return make_random_data_util(n, m)


def make_random_data_util(n, m):
    rnd = np.random.RandomState(SEED)
    data = rnd.randn(n, m) + 5.0 * np.ones((n, m))
    x = np.ma.masked_array(data)
    mean = np.mean(data, axis=0)
    standard_deviation = np.std(data, axis=0)
    return x, mean, standard_deviation


def test_MeanImputer(make_missing_data):
    X = make_missing_data
    Xcopy = X.copy()
    imputer = MeanImputer()
    Ximp = imputer(X)
    assert np.all(~np.isnan(Ximp))
    assert np.all(~np.isinf(Ximp))
    assert Ximp.shape == Xcopy.shape
    assert np.ma.count_masked(Ximp) == 0


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
    imputer = NearestNeighboursImputer(nodes=100)
    Ximp = imputer(X)

    assert np.all(~np.isnan(Ximp))
    assert np.all(~np.isinf(Ximp))
    assert Ximp.shape == Xcopy.shape
    assert np.ma.count_masked(Ximp) == 0


def test_NearestNeighbourImputerValueError():
    rnd = np.random.RandomState(SEED)
    X = np.ma.MaskedArray(data=rnd.randn(100, 3), mask=True)
    imputer = NearestNeighboursImputer(nodes=100, k=5)
    X.mask[:4, :] = False
    with pytest.raises(ValueError):
        imputer(X)


def test_NearestNeighbourImputerRowsCleanValueError():
    rnd = np.random.RandomState(SEED)
    X = np.ma.MaskedArray(data=rnd.randn(100, 3), mask=True)
    imputer = NearestNeighboursImputer(nodes=100, k=5)
    X.mask[:10, :2] = False
    with pytest.raises(ValueError):
        imputer(X)


def test_CentreTransform(make_random_data):
    x, mu, std = make_random_data
    x_expected = x - mu
    center_transformer = CentreTransform()
    x_produced = center_transformer(x)
    assert np.array_equal(x_expected, x_produced)


def test_CentreTransform_caching(make_random_data):
    x, mu, std = make_random_data
    x_copy = x.copy()
    center_transformer = CentreTransform()
    center_transformer(x_copy)
    x_translated = x + 3.0 * mu
    x_expected = x_translated - mu
    x_produced = center_transformer(x_translated)
    assert np.array_equal(x_expected, x_produced)


p_transforms = {np.log: LogTransform,
                np.sqrt: SqrtTransform}


@pytest.fixture(params=list(p_transforms.keys()))
def positive_transform(request):
    return request.param, p_transforms[request.param]


def test_PositiveTransform(make_random_data, positive_transform):
    func, trans = positive_transform
    x, mu, std = make_random_data
    x_expected = func(x - x.min(axis=0) + 1.0e-6)
    sqrt_transformer = trans()
    x_produced = sqrt_transformer(x)
    assert np.array_equal(x_expected, x_produced)


def test_PositiveTransform_caching(make_random_data, positive_transform):
    func, trans = positive_transform
    x, mu, std = make_random_data
    x_copy = x.copy()
    stabilizer = 1.0e-6
    sqrt_transformer = trans(stabilizer=stabilizer)
    sqrt_transformer(x_copy)
    x_translated = x + 15
    x_expected = func(x_translated - x.min(axis=0) + stabilizer)
    x_produced = sqrt_transformer(x_translated)
    assert np.array_equal(x_expected, x_produced)


def test_StandardiseTransform(make_random_data):
    x, mu, std = make_random_data
    x_expected = (x - mu) / std
    standardiser = StandardiseTransform()
    x_produced = standardiser(x)
    assert np.array_equal(x_expected, x_produced)


def test_StandardiseTransform_caching(make_random_data):
    x, mu, std = make_random_data
    x_copy = x.copy()
    center_transformer = CentreTransform()
    center_transformer(x_copy)
    x_translated = x + 3.0 * mu
    x_expected = x_translated - mu
    x_produced = center_transformer(x_translated)
    assert np.array_equal(x_expected, x_produced)


def test_WhitenTransform(make_random_data):
    x, mu, std = make_random_data
    whitener = WhitenTransform(1.0)
    x_produced = whitener(x)
    x_cov = np.cov(x_produced, rowvar=False)
    column_products = abs(x_cov[~np.eye(len(x_cov), dtype=bool)])
    assert np.all(np.less(column_products, 1e-5))


def test_WhitenTransform_caching():
    x, mu, std = make_random_data_util(6, 3)
    x = (x - mu) / std
    x_zero = np.ma.array(np.vstack([np.eye(3), -np.eye(3)])) * (2 ** (-1/2))
    whitener = WhitenTransform(1.0)
    x_zero_white = whitener(x_zero)
    trans = x_zero_white.dot( np.linalg.pinv(x_zero) )
    x_produced = whitener(x)
    x_trans = trans.dot(x_produced)
    x_expected = trans.dot(x)
    assert np.all(np.equal(x_trans, x_expected))
