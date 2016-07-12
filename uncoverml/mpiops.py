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
        self.root_chunk = self.true_ids[0]

    def __enter__(self):
        self.dcomm = comm.Split(self.flag, self.new_index)

    def __exit__(self):
        self.dcomm.Free()


def run_if(f, flag, *args, **kwargs):
    with PredicateGroup(flag) as p:
        if flag:
            kwargs.update({"comm": p.dcomm})
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
    x_sets = run_if(_compute_unique, flag, max_onehot_dims=max_onehot_dims)
    return x_sets


def run_once(f, *args, **kwargs):
    if chunk_index == 0:
        f_result = f(*args, **kwargs)
    result = comm.bcast(f_result, root=0)
    return result
