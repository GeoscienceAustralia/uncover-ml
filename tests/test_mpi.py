from uncoverml import mpiops


def test_helloworld():
    comm = mpiops.comm
    ranks = comm.allgather(mpiops.chunk_index)
    assert len(ranks) == mpiops.chunks
