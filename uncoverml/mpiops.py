import logging
import numpy as np
from mpi4py import MPI
from uncoverml import stats

log = logging.getLogger(__name__)

# MPI globals
comm = MPI.COMM_WORLD
chunks = comm.Get_size()
chunk_index = comm.Get_rank()


class PredicateGroup:
    def __init__(self, flag):
        self.flag = flag
        flag_mask = comm.allgather(flag)
        flags_with_ids = enumerate(flag_mask)
        true_ids = [i for i, v in flags_with_ids if v]
        false_ids = [i for i, v in flags_with_ids if not v]
        i = chunk_index
        self.new_index = (true_ids.index(i) if flag_mask[i] else
                          (-1 * false_ids.index(i) - 1))
        self.root_chunk = true_ids[0]

    def __enter__(self):
        self.dcomm = comm.Split(self.flag, self.new_index)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.dcomm.Free()


def run_if(f, flag, *args, **kwargs):
    with PredicateGroup(flag) as p:
        if flag:
            if kwargs:
                kwargs.update({"comm": p.dcomm})
            else:
                kwargs = {"comm": p.dcomm}

            f_result = f(*args, **kwargs)
        result = comm.bcast(f_result, root=p.root_chunk)
    return result


def _compute_unique(x, comm, max_onehot_dims):
    x_sets = None
    # check data is okay
    if x.dtype == np.dtype('float32') or x.dtype == np.dtype('float64'):
        log.warn("Cannot use one-hot for floating point data -- ignoring")
    else:
        local_sets = stats.sets(x)
        unique_op = MPI.Op.Create(stats.unique, commute=True)
        full_sets = comm.allreduce(local_sets, op=unique_op)
        total_dims = np.sum([len(k) for k in full_sets])
        log.info("Total features from one-hot encoding: {}".format(
            total_dims))
        if total_dims <= max_onehot_dims:
            x_sets = full_sets
        else:
            log.warn("Too many distinct values for one-hot encoding.")
    return x_sets


def compute_unique_values(x, max_onehot_dims):
    flag = x is not None
    x_sets = run_if(_compute_unique, flag, x=x,
                    max_onehot_dims=max_onehot_dims)
    return x_sets


def run_once(f, *args, **kwargs):
    if chunk_index == 0:
        f_result = f(*args, **kwargs)
    else:
        f_result = None
    result = comm.bcast(f_result, root=0)
    return result


def sum_axis_0(x, y, dtype):
    s = np.sum(np.array([x, y]), axis=0)
    return s

sum0_op = MPI.Op.Create(sum_axis_0, commute=True)


def compose_transform(x, settings):
    flag = x is not None
    x_sets = run_if(_compose_transform, flag, x=x,
                    settings=settings)
    return x_sets


def _compose_transform(x, settings, comm):

    x_n_local = stats.count(x)
    x_n = comm.allreduce(x_n_local, op=MPI.SUM)
    x_full_local = stats.full_count(x)
    x_full = comm.allreduce(x_full_local, op=MPI.SUM)

    out_dims = x_n.shape[0]
    log.info("Total input dimensionality: {}".format(x_n.shape[0]))
    fraction_missing = (1.0 - np.sum(x_n) / (x_full * x_n.shape[0])) * 100.0
    log.info("Input data is {}% missing".format(fraction_missing))

    if settings.impute:
        if settings.impute_mean is None:
            local_impute_sum = stats.sum(x)
            impute_sum = comm.allreduce(local_impute_sum, op=sum0_op)
            impute_mean = impute_sum / x_n
            log.info("Imputing missing data from mean {}".format(impute_mean))
            settings.imput_mean = impute_mean
        impute_mean = settings.mean
        stats.impute_with_mean(x, impute_mean)
        x_n_local = stats.count(x)
        x_n = comm.allreduce(x_n_local, op=MPI.SUM)

    if settings.transform in {"centre", "standardise", "whiten"}:
        if settings.mean is None:
            x_sum_local = stats.sum(x)
            x_sum = comm.allreduce(x_sum_local, op=sum0_op)
            mean = x_sum / x_n
            settings.mean = mean

        log.info("Subtracting global mean {}".format(mean))
        stats.centre(x.data, mean)
        mean = np.zeros_like(mean)

    if settings.transform in {"standardise", "whiten"}:
        if settings.sd is None:
            x_var_local = stats.var(x, mean)
            x_var = comm.allreduce(x_var_local, op=sum0_op)
            sd = np.sqrt(x_var / x_n)
            settings.sd = sd

        log.info("Dividing through global standard deviation {}".format(sd))
        stats.standardise(x.data, sd, mean)
        mean = np.zeros_like(mean)

    if settings.transform == "whiten":
        if not settings.eigvals or not settings.eigvecs:
            x_outer_local = stats.outer(x, mean)
            outer = comm.allreduce(x_outer_local, op=sum0_op)
            cov = outer / x_n
            eigvals, eigvecs = np.linalg.eigh(cov)
            settings.eigvals = eigvals
            settings.eigvecs = eigvecs

        out_dims = int(out_dims * settings.featurefraction)
        log.info("Whitening and keeping {} dimensions".format(out_dims))
        ndims = x.shape[1]
        # make sure 1 <= keepdims <= ndims
        keepdims = min(max(1, int(ndims * settings.featurefraction)), ndims)
        mat = eigvecs[:, -keepdims:]
        vec = eigvals[-keepdims:]
        x = np.ma.dot(x, mat, strict=True) / np.sqrt(vec)

    return x, settings
