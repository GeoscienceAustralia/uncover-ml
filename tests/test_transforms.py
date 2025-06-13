from scipy.stats import bernoulli

from uncoverml.transforms.onehot import (sets, compute_unique_values, one_hot,
                                        OneHotTransform, RandomHotTransform)
from uncoverml.transforms.impute import (GaussImputer,
                                         NearestNeighboursImputer, MeanImputer)
from uncoverml.transforms.linear import (CentreTransform, StandardiseTransform,
                                         WhitenTransform, LogTransform,
                                         SqrtTransform)
from uncoverml.transforms.transformset import (build_feature_vector, missing_percentage,
                                            TransformSet, ImageTransformSet)
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


def test_compute_unique_values_on_int_array():
    x = np.ma.array([[1, 2], [3, 4], [1, 2]])
    expected = [np.array([1, 3]), np.array([2, 4])]
    result = compute_unique_values(x)
    assert len(result) == 2
    assert all(np.array_equal(a, b) for a, b in zip(result, expected))


def test_compute_unique_values_raises_on_float():
    x = np.ma.array([[1.0, 2.0], [3.0, 4.0]], dtype='float64')
    with pytest.raises(ValueError, match="Can't do one-hot on float data"):
        compute_unique_values(x)


def test_one_hot_simple():
    x = np.ma.array([[[[0], [1]]]], mask=False)
    x_set = [np.array([0, 1])]
    result = one_hot(x, x_set)
    expected = np.array([[[[0.5, 0. ], [0. , 0.5]]]])
    np.testing.assert_array_equal(result.data, expected)
    assert not np.any(result.mask)


def test_one_hot_with_mask():
    x = np.ma.array([[[[0], [1]]]], mask=[[[[False], [True]]]])
    x_set = [np.array([0, 1])]
    result = one_hot(x, x_set)
    assert result.mask.shape == (1, 1, 2, 2)
    assert result.mask[0, 0, 0, 0] == False
    assert result.mask[0, 0, 1, 0] == True
    assert isinstance(result, np.ma.MaskedArray)


def test_onehot_transform():
    x = np.ma.array(np.random.randint(0, 3, size=(2, 2, 2, 2)))
    transformer = OneHotTransform()
    result = transformer(x)
    assert isinstance(result, np.ma.MaskedArray)


def test_randomhot_transform():
    x = np.ma.array(np.random.randint(0, 2, size=(2, 2, 2, 2)))
    transformer = RandomHotTransform(n_features=4, seed=SEED)
    result = transformer(x)
    assert isinstance(result, np.ma.MaskedArray)
    assert result.shape[3] == 8


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


def test_build_feature_vector():
    data1 = np.ma.masked_array(data=np.ones((3, 2, 1), dtype=int), mask=np.zeros((3, 2, 1)))
    data2 = np.ma.masked_array(data=3 * np.ones((3, 2, 1), dtype=int), mask=np.zeros((3, 2, 1)))
    image_chunks = {'x': data1.copy(), 'y': data2.copy()}
    x = build_feature_vector(image_chunks, is_categorical=True)
    assert x.shape == (3, 4)
    assert np.all(x.data[:, :2] == 1)
    assert np.all(x.data[:, 2:] == 3)


def test_missing_percentage():
    mask = np.zeros((10, 5), dtype=bool)
    mask[0, 0] = True
    x = np.ma.masked_array(data=np.ones((10, 5)), mask=mask)
    result = missing_percentage(x)
    assert 0.0 < result < 100.0


def test_TransformSet():
    class DummyImputer:
        def __call__(self, x):
            return x * 2

    class DummyTransform:
        def __call__(self, x):
            return x + 1

    x = np.ma.masked_array(data=np.ones((5, 5)), mask=np.zeros((5, 5)))
    ts = TransformSet(imputer=DummyImputer(), transforms=[DummyTransform()])
    result = ts(x)
    expected = (x * 2) + 1
    assert np.all(result == expected)


def test_ImageTransformSet():
    class DummyImageTransform:
        def __init__(self, factor):
            self.factor = factor

        def __call__(self, x):
            return x * self.factor

    class DummyGlobalTransform:
        def __call__(self, x):
            return x + 5

    image1 = np.ma.masked_array(data=np.ones((2, 2, 2)), mask=np.zeros((2, 2, 2)))
    image2 = np.ma.masked_array(data=2 * np.ones((2, 2, 2)), mask=np.zeros((2, 2, 2)))
    image_chunks = {
        'a': image1,
        'b': image2
    }

    img_trans = [DummyImageTransform(10), DummyImageTransform(100)]
    global_trans = [DummyGlobalTransform()]
    ts = ImageTransformSet(image_transforms=[img_trans], global_transforms=global_trans)
    result = ts(image_chunks)
    assert result.shape == (2, 8)
    assert np.allclose(result.data[:, :4], 10.0 + 5)
    assert np.allclose(result.data[:, 4:], 200.0 + 5)
