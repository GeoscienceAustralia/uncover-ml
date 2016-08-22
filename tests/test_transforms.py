from uncoverml.transforms.onehot import sets, one_hot
from uncoverml.transforms.impute import impute_with_mean
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
