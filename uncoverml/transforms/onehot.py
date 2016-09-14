import logging

import numpy as np

from uncoverml import mpiops

log = logging.getLogger(__name__)


def sets(x):
    """
    works on a masked x
    """
    sets = [np.unique(np.ma.compressed(x[:, i])) for i in range(x.shape[1])]
    return sets


def compute_unique_values(x):
    """compute per-dimension unique values over a data vector

    This function computes the set of unique values for each dimension in
    x, unless the number of unique values exceeds max_onehot_dims.

    Parameters
    ----------
    x : ndarray (n x m)
        The array over which to compute unique values. The set is over
        the first dimension

    Returns
    -------
    x_sets : list of ndarray or None
        A list of m sets of unique values for each dimension in x
    """
    # check data is okay
    if x.dtype == np.dtype('float32') or x.dtype == np.dtype('float64'):
        raise ValueError("Can't do one-hot on float data")
    else:
        local_sets = sets(x)
        full_sets = mpiops.comm.allreduce(local_sets, op=mpiops.unique_op)
    return full_sets


def one_hot(x, x_set, matrices=None):
    assert x.ndim == 4  # points, patch_x, patch_y, channel
    if matrices:
        out_dim_sizes = np.array([m.shape[1] for m in matrices])
    else:
        out_dim_sizes = np.array([k.shape[0] for k in x_set])
    # The index points in the output array for each input dimension
    indices = np.hstack((np.array([0]), np.cumsum(out_dim_sizes)))
    total_dims = np.sum(out_dim_sizes)
    out_shape = x.shape[0:3] + (total_dims,)
    out = np.zeros(out_shape, dtype=float)

    for dim_idx, dim_set in enumerate(x_set):
        # input data
        dim_in = x.data[..., dim_idx]

        # appropriate parts of the output_array
        dim_out = out[..., indices[dim_idx]:indices[dim_idx + 1]]

        # compute the one-hot values
        for i, val in enumerate(dim_set):
            if matrices:
                proj = matrices[dim_idx]
                dim_out[dim_in == val] = proj[i]
            else:
                dim_out[..., i][dim_in == val] = 0.5

    if x.mask.ndim != 0:  # all false
        out_mask = np.zeros(out_shape, dtype=bool)
        for dim_idx, dim_set in enumerate(x_set):
            dim_mask = x.mask[..., dim_idx]
            dim_out_mask = out_mask[..., indices[dim_idx]:indices[dim_idx + 1]]
            # broadcast the mask
            dim_out_mask[:] = dim_mask[..., np.newaxis]
    else:
        out_mask = False

    result = np.ma.MaskedArray(data=out, mask=out_mask)
    return result


class OneHotTransform:
    def __init__(self):
        self.x_sets = None

    def __call__(self, x):
        x = x.astype(int)
        if self.x_sets is None:
            self.x_sets = compute_unique_values(x)

        for s in self.x_sets:
            log.info("One-hot encoding to d={}".format(len(s)))
        x = one_hot(x, self.x_sets)
        return x


class RandomHotTransform:
    def __init__(self, n_features, seed):
        self.n_features = n_features
        self.seed = seed
        self.matrices = None

    def __call__(self, x):
        x = x.astype(int)
        if self.matrices is None:
            np.random.seed(self.seed)
            self.x_sets = compute_unique_values(x)
            nbands = [len(s) for s in self.x_sets]
            self.matrices = [np.random.randn(k, self.n_features)
                             for k in nbands]
        for s in self.x_sets:
            log.info("One-hot encoding to "
                     "d={} space then projecting to d={}".format(
                         len(s), self.n_features))
        x = one_hot(x, self.x_sets, self.matrices)
        return x
