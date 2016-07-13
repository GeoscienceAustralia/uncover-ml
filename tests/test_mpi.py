import pytest
from uncoverml import mpiops


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
