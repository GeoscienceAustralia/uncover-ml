from uncoverml import parallel
from collections import namedtuple
import numpy as np

def test_direct_view(make_ipcluster):
    cluster = parallel.direct_view(profile=None,nchunks=10)
    cluster.execute("import sys; mods = set(sys.modules.keys())")
    for i in range(len(cluster)):
        assert 'uncoverml.feature' in cluster['mods'][i]
        assert 'uncoverml.parallel' in cluster['mods'][i]
        assert 'numpy' in cluster['mods'][i]
    #check chunks
    chunk_indices = [[0,1,2],[3,4,5],[6,7],[8,9]]
    assert cluster["chunk_indices"] == chunk_indices

class FakeImage:
    def __init__(self, x, m):
        Img = namedtuple("MaskedArray", ['data','mask'])
        self.img = Img(data=x,mask=m)
    def data(self):
        return self.img

def test_node_count(masked_array):
    r = parallel.node_count(masked_array)
    assert np.all(r == np.array([2,2],dtype=int))

def test_node_sum(masked_array):
    r = parallel.node_sum(masked_array)
    assert np.all(r == np.array([6.0, 6.0]))

def test_node_var(masked_array):
    r = parallel.node_var(masked_array)
    assert np.all(r == np.array([2.0, 8.0]))

def test_node_outer():
    x = np.ma.masked_array(data=[[0.,1.],[2.,3.],[4.,5.], [3.,3.]],
                           mask=[[True,False,],[False,True],[False,False],
                                 [False,False]],
                           fill_value=1e23)
    r = parallel.node_outer(x)
    assert np.all(r == np.array([[2.,2.],[2.,8.]]))

def test_node_sets(int_masked_array):
    r = parallel.node_sets(int_masked_array)
    assert(np.all(r == np.array([[2,3,4],[1,3,5]],dtype=int)))


def test_one_hot(int_masked_array):
    x_set = np.array([[2,3,4],[1,3,5]],dtype=int)
    r = parallel.one_hot(int_masked_array, x_set)
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





