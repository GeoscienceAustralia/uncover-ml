import pytest
from uncoverml import mpiops


# Make sure all MPI tests use this fixure
@pytest.fixture()
def mpisync(request):
    mpiops.comm.barrier()

    def fin():
        mpiops.comm.barrier()
    request.addfinalizer(fin)
    return mpiops.comm


def test_helloworld(mpisync):
    comm = mpiops.comm
    ranks = comm.allgather(mpiops.chunk_index)
    assert len(ranks) == mpiops.chunks


def test_run_if(mpisync):
    idx = mpiops.chunk_index

    def f(x, comm):
        return x + comm.rank  # this is rank in the split comm

    flag = idx != 0
    result = mpiops.run_if(f, flag, x=0)
    true_result = None if idx == 0 else idx - 1
    assert result == true_result


def test_run_if_broadcast(mpisync):
    idx = mpiops.chunk_index

    def f(x, comm):
        return x + comm.rank  # this is rank in the split comm

    flag = idx != 0
    result = mpiops.run_if(f, flag, x=0, broadcast=True)
    true_result = 0
    assert result == true_result
