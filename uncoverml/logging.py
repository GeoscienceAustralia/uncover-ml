import logging
import resource
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
    my_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    total_usage = mpiops.comm.reduce(my_usage, root=0)
    return total_usage


class MPIStreamHandler(logging.StreamHandler):
    """
    Only logs messages from Node 0
    """
    def emit(self, record):
        total_usage = _total_gb()
        if mpiops.chunk_index == 0:
            record.mem = total_usage
            super().emit(record)


class ElapsedFormatter():

    def format(self, record):
        lvl = record.levelname
        name = record.name
        t = int(round(record.relativeCreated/1000.0))
        msg = record.getMessage()
        return "+{}s {:.2f}GB {}:{} {}".format(t, record.mem, name, lvl, msg)
