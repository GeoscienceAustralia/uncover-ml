import logging
from mpi4py import MPI
from uncoverml import ipympi


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    log_format = ("node{} %(threadName)s %(relativeCreated)6d " +
                  "%(levelname)s: %(message)s").format(rank)
    log_level = logging.INFO
    log_to_files = False
    if log_to_files:
        logging.basicConfig(level=log_level,
                            format=log_format,
                            filename='ipympi_node{}.log'.format(rank))
    else:
        logging.basicConfig(level=log_level,
                            format=log_format)

    def fn():
        print("Hello world!")

    ipympi.call_with_ipympi(fn)
