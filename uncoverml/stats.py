
import numpy as np


# def count(x):
#     """
#     note that this is a vector per dimension because x is masked
#     """
#     return np.ma.count(x, axis=0)


# def full_count(x):
#     """
#     total number of points including missing
#     """
#     return x.shape[0]


# def sum(x):
#     s = np.ma.sum(x, axis=0)
#     if np.ma.count_masked(s) != 0:
#         raise ValueError("Too many missing values to compute sum")
#     result = s.data
#     return result


# def outer(x, mean):
#     delta = x - mean
#     result = np.ma.dot(delta.T, delta)
#     if np.ma.count_masked(result) != 0:
#         raise ValueError("Too many missing values to compute outer product")
#     return result.data


def sets(x):
    """
    works on a masked x
    """
    sets = [np.unique(np.ma.compressed(x[:, i])) for i in range(x.shape[1])]
    return sets


# def centre(x, mean):
#     x -= mean


# def standardise(x, x_sd, mean):
#     centre(x, mean)
#     x /= x_sd


def impute_with_mean(x, mean):

    # No missing data
    if np.ma.count_masked(x) == 0:
        return

    for i, r in enumerate(x):
        x.data[i][x.mask[i]] = mean[x.mask[i]]
    x.mask *= False


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


def unique(sets1, sets2, dtype):
    per_dim = zip(sets1, sets2)
    out_sets = [np.unique(np.concatenate(k, axis=0)) for k in per_dim]
    return out_sets


