import logging

import numpy as np

from uncoverml import mpiops
from uncoverml import defaults as df

log = logging.getLogger(__name__)


def sets(x):
    """
    works on a masked x
    """
    sets = [np.unique(np.ma.compressed(x[:, i])) for i in range(x.shape[1])]
    return sets


def compute_unique_values(x, max_onehot_dims):
    """compute per-dimension unique values over a data vector

    This function computes the set of unique values for each dimension in
    x, unless the number of unique values exceeds max_onehot_dims.

    Parameters
    ----------
    x : ndarray (n x m)
        The array over which to compute unique values. The set is over
        the first dimension
    max_onehot_dims : int
        The maximum number of unique values to accept. If exceeded returns
        None

    Returns
    -------
    x_sets : list of ndarray or None
        A list of m sets of unique values for each dimension in x
    """
    x_sets = None
    # check data is okay
    if x.dtype == np.dtype('float32') or x.dtype == np.dtype('float64'):
        log.warn("Cannot use one-hot for floating point data -- ignoring")
    else:
        local_sets = sets(x)
        full_sets = mpiops.comm.allreduce(local_sets, op=mpiops.unique_op)
        total_dims = np.sum([len(k) for k in full_sets])
        log.info("Total features from one-hot encoding: {}".format(
            total_dims))
        if total_dims <= max_onehot_dims:
            x_sets = full_sets
        else:
            log.warn("Too many distinct values for one-hot encoding.")
    return x_sets


def one_hot(x, x_set):
    assert x.data.shape == x.mask.shape
    out_dim_sizes = np.array([k.shape[0] for k in x_set])
    # The index points in the output array for each input dimension
    indices = np.hstack((np.array([0]), np.cumsum(out_dim_sizes)))
    total_dims = np.sum(out_dim_sizes)
    n = x.shape[0]
    out = np.empty((n, total_dims), dtype=float)
    out.fill(-0.5)
    out_mask = np.zeros((n, total_dims), dtype=bool)

    for dim_idx, dim_set in enumerate(x_set):
        dim_in = x[:, dim_idx]
        dim_mask = x.mask[:, dim_idx]
        dim_out = out[:, indices[dim_idx]:indices[dim_idx + 1]]
        dim_out_mask = out_mask[:, indices[dim_idx]:indices[dim_idx + 1]]
        dim_out_mask[:] = dim_mask[:, np.newaxis]
        for i, val in enumerate(dim_set):
            dim_out[:, i][dim_in == val] = 0.5
    result = np.ma.array(data=out, mask=out_mask)
    return result


class OneHotTransform:
    def __init__(self):
        self.x_sets = None

    def __call__(self, x):
        if self.x_sets is None:
            self.x_sets = compute_unique_values(x, df.max_onehot_dims)
        x = one_hot(x, self.x_sets)
        return x
