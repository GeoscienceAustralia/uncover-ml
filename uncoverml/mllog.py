import logging
import sys
import traceback
import warnings

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
    If message stars with ':mpi:', the message will be logged 
    regardless of node (the ':mpi:' will be removed from the message).
    Otherwise, only node 0 will emit messages.
    """
    def emit(self, record):
        if record.msg.startswith(':mpi:'):
            record.msg = record.msg.replace(':mpi:', '')
            super().emit(record)
        elif mpiops.chunk_index == 0:
            super().emit(record)


class ElapsedFormatter():

    def format(self, record):
        lvl = record.levelname
        name = record.name
        t = int(round(record.relativeCreated/1000.0))
        msg = record.getMessage()
        logstr = "+{}s {} {} [P{}]: {}".format(t, lvl, name, mpiops.chunk_index, msg)
        return logstr


def warn_with_traceback(message, category, filename, lineno, line=None):
    """
    copied from:
    http://stackoverflow.com/questions/22373927/get-traceback-of-warnings
    """
    traceback.print_stack()
    log = sys.stderr
    log.write(warnings.formatwarning(
        message, category, filename, lineno, line))
