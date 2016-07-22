from uncoverml import stats
import numpy as np


def test_sets(int_masked_array):
    r = stats.sets(int_masked_array)
    assert(np.all(r == np.array([[2, 3, 4], [1, 3, 5]], dtype=int)))


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
