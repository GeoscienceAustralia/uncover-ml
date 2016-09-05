import logging
from uncoverml import mpiops


def configure(verbosity):
    log = logging.getLogger("")
    log.setLevel(verbosity)
    ch = MPIStreamHandler()
    formatter = ElapsedFormatter()
    ch.setFormatter(formatter)
    log.addHandler(ch)


class MPIStreamHandler(logging.StreamHandler):
    """
    Only logs messages from Node 0
    """
    def emit(self, record):
        if mpiops.chunk_index == 0:
            super().emit(record)


class ElapsedFormatter():

    def format(self, record):
        lvl = record.levelname
        name = record.name
        t = int(round(record.relativeCreated/1000.0))
        msg = record.getMessage()
        return "+{}s {}:{} {}".format(t, name, lvl, msg)
