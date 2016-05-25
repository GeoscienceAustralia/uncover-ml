
import numpy as np


def count(x):
    """
    note that this is a vector per dimension because x is masked
    """
    return x.count(axis=0)


def full_count(x):
    """
    total number of points including missing
    """
    return x.shape[0]


def sum(x):
    result = np.ma.sum(x, axis=0)
    if np.ma.count_masked(result) != 0:
        raise ValueError("Too many missing values to compute sum")
    return result


def var(x):
    delta = x - np.ma.mean(x, axis=0)
    result = np.ma.sum(delta * delta, axis=0)
    if np.ma.count_masked(result) != 0:
        raise ValueError("Too many missing values to compute variance")
    return result.data


def outer(x):
    delta = x - np.ma.mean(x, axis=0)
    result = np.ma.dot(delta.T, delta)
    if np.ma.count_masked(result) != 0:
        raise ValueError("Too many missing values to compute outer product")
    return result.data


def sets(x):
    """
    works on a masked x
    """
    sets = [np.unique(np.ma.compressed(x[:, i])) for i in range(x.shape[1])]
    return sets


def centre(x, x_mean):
    return x - x_mean


def standardise(x, x_sd):
    return x / x_sd[np.newaxis, :]


def impute_with_mean(x, mean):
    xi = np.ma.masked_array(data=np.copy(x.data), mask=False)
    xi.data[x.mask] = np.broadcast_to(mean, x.shape)[x.mask]
    return xi


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
        dim_out = out[:, indices[dim_idx]:indices[dim_idx+1]]
        dim_out_mask = out_mask[:, indices[dim_idx]:indices[dim_idx+1]]
        dim_out_mask[:] = dim_mask[:, np.newaxis]
        for i, val in enumerate(dim_set):
            dim_out[:, i][dim_in == val] = 0.5
    result = np.ma.array(data=out, mask=out_mask)
    return result
        
