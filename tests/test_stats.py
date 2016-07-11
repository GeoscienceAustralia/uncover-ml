from uncoverml import stats
from collections import namedtuple
import numpy as np


def test_count(masked_array):
    r = stats.count(masked_array)
    assert np.all(r == np.array([2,  2],  dtype=int))


def test_full_count(masked_array):
    r = stats.full_count(masked_array)
    assert r == 3


def test_sum(masked_array):
    r = stats.sum(masked_array)
    assert np.all(r == np.array([6.0,  6.0]))


def test_var(masked_array):
    mean = np.array([3.0, 3.0])
    r = stats.var(masked_array, mean)
    assert np.all(r == np.array([2.0,  8.0]))


def test_outer():
    x = np.ma.masked_array(data=[[0., 1.], [2., 3.], [4., 5.],  [3., 3.]],
                           mask=[[True, False, ],
                                 [False, True],
                                 [False, False],
                                 [False, False]],
                           fill_value=1e23)
    r = stats.outer(x, mean=np.ma.mean(x, axis=0))
    assert np.all(r == np.array([[2., 2.], [2., 8.]]))


def test_sets(int_masked_array):
    r = stats.sets(int_masked_array)
    assert(np.all(r == np.array([[2, 3, 4], [1, 3, 5]], dtype=int)))


def test_centre(masked_array):
    t = np.copy(masked_array)
    stats.centre(t, mean=np.ma.mean(t, axis=0))
    assert np.all(np.ma.mean(t, axis=0) == 0)


def test_standardise(masked_array):
    t = np.copy(masked_array)
    mean = np.ma.mean(t, axis=0)
    sd = np.ma.std(t, axis=0)
    stats.standardise(t, sd, mean)
    assert np.allclose(np.ma.std(t, axis=0), 1.0)


def test_impute_with_mean(masked_array):
    t = np.ma.copy(masked_array)
    mean = np.ma.mean(t, axis=0)
    trueval = np.array([[3.0,  1.0], [2.0, 3.0], [4.0, 5.0]])
    stats.impute_with_mean(t, mean)
    assert np.all(t.data == trueval)
    assert np.all(np.logical_not(t.mask))


def test_one_hot(int_masked_array):
    x_set = np.array([[2, 3, 4], [1, 3, 5]], dtype=int)
    r = stats.one_hot(int_masked_array,  x_set)
    ans = np.array([[-0.5,  -0.5,  -0.5,   0.5,  -0.5,  -0.5],
                   [0.5,  -0.5,  -0.5,  -0.5,  -0.5,  -0.5],
                   [-0.5,  -0.5,   0.5,  -0.5,  -0.5,   0.5],
                   [-0.5,   0.5,  -0.5,  -0.5,   0.5,  -0.5]])
    ans_mask = np.array(
        [[True,   True,   True,  False,  False,  False],
         [False,  False,  False,   True,   True,   True],
         [False,  False,  False,  False,  False,  False],
         [False,  False,  False,  False,  False,  False]],  dtype=bool)
    assert np.all(r.data == ans)
    assert np.all(r.mask == ans_mask)
