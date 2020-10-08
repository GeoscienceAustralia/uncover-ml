import logging
import pickle
import inspect

import numpy as np
from mpi4py import MPI

_logger = logging.getLogger(__name__)


comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()
leader_world = rank_world == 0

# Determine which node each rank is on and share this information
node_name = MPI.Get_processor_name()
node_names = comm_world.allgather(node_name)
root_node = None
# Map node names to the local leader of that node's sub-group
node_map = {}
prev_node = None
for i, n in enumerate(node_names):
    if i == 0:
        node_map[n] = 0
        prev_node = n
    elif prev_node != n:
        node_map[n] = i

    prev_node = n

# Split communicator into sub-groups on each node 
# This is so each group is on a physical node capable of sharing a memory window 
comm_local = MPI.Comm.Split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED)
size_local = comm_local.Get_size()
rank_local = comm_local.Get_rank()
leader_local = rank_local == 0

# Test data transfer between subgroups
if rank_world == 0:
    x = 1
else:
    x = None

print("X before sharing")

if leader_local:
    print(f"World rank: {rank_world}, x: {x}")

print("Sharing data with local leaders")

for k, v in node_map.items():
    if leader_world:
        comm_world.send(x, dest=v, tag=99)
    elif leader_local and not leader_world:
        x = comm_world.recv(source=leader_world, tag=99)

if leader_local:
    print(f"World rank: {rank_world}, x: {x}")


def create_shared_array(data, root=0, writeable=False):
    """
    Create a shared numpy array among MPI nodes. To access the data,
    refer to the return numpy array 'shared'. The second return value
    is the MPI window. This doesn't need to be interacted with except
    when deallocating the memory.

    When finished with the data, set `shared = None` and call 
    `win.Free()`.

    Caution: any node with a handle on the shared array can modify its
    contents. To be safe, the shared array is set to read-only by 
    default.

    Parameters
    ----------
    data : numpy.ndarray
        The numpy array to share.
    root : int
        Rank of the root node that contains the original data.
    writeable : bool
        Whether or not the resulting shared array is writeable.

    Returns
    -------
    tuple of numpy.ndarray, MPI window
    """
    if chunk_index == root:
        shape = data.shape
        dtype = data.dtype
        item_size = dtype.itemsize
        size = np.prod(shape) * item_size
    else:
        shape = None
        dtype = None
        item_size = 1
        size = 1

    comm.barrier()

    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    win = MPI.Win.Allocate_shared(size, item_size, comm=comm)

    buf, _ = win.Shared_query(root)
    shared = np.ndarray(buffer=buf, dtype=dtype, shape=shape)

    # Copy data into shared arrays - is there a better way than 
    #  iterating the whole thing?
    if chunk_index == root:
        for index, x in np.ndenumerate(data):
            shared[index] = x
        shared.flags.writeable = writeable

    comm.barrier()

    # Make sure to call `shared = None` and `win.free()` to deallocate 
    #  the shared memory when done with it.
    return shared, win        

def run_once(f, *args, **kwargs):
    """Run a function on one node, broadcast result to all
    This function evaluates a function on a single node in the MPI world,
    then broadcasts the result of that function to every node in the world.
    Parameters
    ----------
    f : callable
        The function to be evaluated. Can take arbitrary arguments and return
        anything or nothing
    args : optional
        Other positional arguments to pass on to f
    kwargs : optional
        Other named arguments to pass on to f
    Returns
    -------
    result
        The value returned by f
    """
    if chunk_index == 0:
        f_result = f(*args, **kwargs)
    else:
        f_result = None
    result = comm.bcast(f_result, root=0)
    return result


def sum_axis_0(x, y, dtype):
    s = np.ma.sum(np.ma.vstack((x, y)), axis=0)
    return s


def max_axis_0(x, y, dtype):
    s = np.amax(np.array([x, y]), axis=0)
    return s


def min_axis_0(x, y, dtype):
    s = np.amin(np.array([x, y]), axis=0)
    return s


def unique(sets1, sets2, dtype):
    per_dim = zip(sets1, sets2)
    out_sets = [np.unique(np.concatenate(k, axis=0)) for k in per_dim]
    return out_sets

unique_op = MPI.Op.Create(unique, commute=True)
sum0_op = MPI.Op.Create(sum_axis_0, commute=True)
max0_op = MPI.Op.Create(max_axis_0, commute=True)
min0_op = MPI.Op.Create(min_axis_0, commute=True)


def count_targets(targets):
    return comm.allreduce(len(targets.positions))


def count(x):
    x_n_local = np.ma.count(x, axis=0).ravel()
    x_n = comm.allreduce(x_n_local, op=sum0_op)
    still_masked = np.ma.count_masked(x_n)
    if still_masked != 0:
        log.info('Reported subcounts: ' + ', '.join([str(s) for s in x_n]))
        raise ValueError("Can't compute count: subcounts are still masked")
    if hasattr(x_n, 'mask'):
        x_n = x_n.data
    return x_n


def outer_count(x):

    xnotmask = (~x.mask).astype(float)
    x_n_outer_local = np.dot(xnotmask.T, xnotmask)
    x_n_outer = comm.allreduce(x_n_outer_local)

    return x_n_outer


def mean(x):
    x_n = count(x)
    x_sum_local = np.ma.sum(x, axis=0)
    x_sum = comm.allreduce(x_sum_local, op=sum0_op)
    still_masked = np.ma.count_masked(x_sum)
    if still_masked != 0:
        log.info('Reported x_sum: ' + ', '.join([str(s) for s in x_sum]))
        raise ValueError("Can't compute mean: At least 1 column has nodata")
    if hasattr(x_sum, 'mask'):
        x_sum = x_sum.data
    mean = x_sum / x_n
    return mean


def minimum(x):
    x_min_local = np.ma.min(x, axis=0)
    x_min = comm.allreduce(x_min_local, op=min0_op)
    still_masked = np.ma.count_masked(x_min)
    if still_masked != 0:
        log.info('Reported x_min: ' + ', '.join([str(s) for s in x_min]))
        raise ValueError("Can't compute mean: At least 1 column has nodata")
    if hasattr(x_min, 'mask'):
        x_min = x_min.data
    return x_min


def sd(x):
    x_mean = mean(x)
    delta_mean = mean(power((x - x_mean), 2))
    sd = np.sqrt(delta_mean)
    return sd


def power(x, exp):
    if np.ma.count_masked(x) == 0:
        return np.ma.masked_array(x.data**2, mask=False)
    m = np.where(~x.mask)
    xe = x[m]
    xe = xe**exp
    result = x.copy()
    result[m] = xe
    return result


def outer(x):
    x_outer_local = np.ma.dot(x.T, x)
    out = comm.allreduce(x_outer_local)
    still_masked = np.ma.count_masked(out)
    if still_masked != 0:
        log.info('Reported out: ' + ', '.join([str(s) for s in out]))
        raise ValueError("Can't compute outer product:"
                         " completely missing columns!")
    if hasattr(out, 'mask'):
        out = out.data
    return out


def covariance(x):
    x_mean = mean(x)
    cov = outer(x - x_mean) / outer_count(x)
    return cov


def eigen_decomposition(x):
    eigvals, eigvecs = np.linalg.eigh(covariance(x))
    return eigvals, eigvecs


def random_full_points(x, Napprox):
    npernode = int(np.round(Napprox / chunks))
    npernode = min(npernode, len(x))  # Make sure the dataset is upper bound

    rinds = np.random.permutation(len(x))  # random choice of indices

    # Get random points per node
    x_p_node = []
    count = 0
    for i in rinds:
        if np.ma.count_masked(x[i]) > 0:
            continue
        if count >= npernode:
            break
        x_p_node.append(x[i])
        count += 1

    # one chunk can have all of one or more covariates masked
    x_p_node = np.vstack(x_p_node) if len(x_p_node) else None

    all_x_p_node = comm.allgather(x_p_node)
    # filter out the None chunks
    filter_all_x_p_node = [x for x in all_x_p_node if x is not None]

    # Gather all random points
    x_p = np.vstack(filter_all_x_p_node)
    return x_p
