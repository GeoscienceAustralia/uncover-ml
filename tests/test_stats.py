from uncoverml import stats
from collections import namedtuple
import numpy as np

def test_count(masked_array):
    r = stats.count(masked_array)
    assert np.all(r == np.array([2,2],dtype=int))

def test_sum(masked_array):
    r = stats.sum(masked_array)
    assert np.all(r == np.array([6.0, 6.0]))

def test_var(masked_array):
    r = stats.var(masked_array)
    assert np.all(r == np.array([2.0, 8.0]))

def test_outer():
    x = np.ma.masked_array(data=[[0.,1.],[2.,3.],[4.,5.], [3.,3.]],
                           mask=[[True,False,],[False,True],[False,False],
                                 [False,False]],
                           fill_value=1e23)
    r = stats.outer(x)
    assert np.all(r == np.array([[2.,2.],[2.,8.]]))

def test_sets(int_masked_array):
    r = stats.sets(int_masked_array)
    assert(np.all(r == np.array([[2,3,4],[1,3,5]],dtype=int)))


def test_one_hot(int_masked_array):
    x_set = np.array([[2,3,4],[1,3,5]],dtype=int)
    r = stats.one_hot(int_masked_array, x_set)
    ans = np.array([[-0.5, -0.5, -0.5,  0.5, -0.5, -0.5],
                   [ 0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
                   [-0.5, -0.5,  0.5, -0.5, -0.5,  0.5],
                   [-0.5,  0.5, -0.5, -0.5,  0.5, -0.5]])
    ans_mask = np.array([[ True,  True,  True, False, False, False],
                   [False, False, False,  True,  True,  True],
                   [False, False, False, False, False, False],
                   [False, False, False, False, False, False]], dtype=bool)
    assert np.all(r.data == ans)
    assert np.all(r.mask == ans_mask)
