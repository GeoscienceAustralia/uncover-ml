import logging
import psutil
import resource
import os
from uncoverml import mpiops


def configure(verbosity):
    log = logging.getLogger("")
    log.setLevel(verbosity)
    ch = MPIStreamHandler()
    formatter = ElapsedFormatter()
    ch.setFormatter(formatter)
    log.addHandler(ch)


def _total_gb():
    # given in KB so convert
    my_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    total_usage = mpiops.comm.reduce(my_usage, root=0)
    return total_usage


def _current_gb():
    process = psutil.Process(os.getpid())
    # in bytes
    my_mem = process.memory_full_info().uss / (1024**3)
    total_gb = mpiops.comm.reduce(my_mem, root=0)
    return total_gb


class MPIStreamHandler(logging.StreamHandler):
    """
    Only logs messages from Node 0
    """
    def emit(self, record):
        total_usage = _total_gb()
        current_usage = _current_gb()
        if mpiops.chunk_index == 0:
            record.mem_total = total_usage
            record.mem_current = current_usage
            super().emit(record)


class ElapsedFormatter():

    def format(self, record):
        lvl = record.levelname
        name = record.name
        t = int(round(record.relativeCreated/1000.0))
        msg = record.getMessage()
        memc = record.mem_current
        memt = record.mem_total
        return "+{}s {:.1f}GB/{:.1f}GB {}:{} {}".format(t, memc, memt,
                                                        name, lvl, msg)
