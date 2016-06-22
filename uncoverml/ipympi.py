import logging
import select
import threading
import time
import subprocess

import numpy as np
from mpi4py import MPI

# Globals from MPI and logging
log = logging.getLogger(__name__)


def sync_to_node(rank, comm):
    """
    All nodes in mpi cluster wait for node with rank
    """
    flag = np.zeros(1)
    flag = comm.bcast(flag, root=rank)


def call_with_ipympi(fn):
    """
    Call a function ensuring an ipyparallel cluster is ready

    Spin up a cluster, wait until it is ready, call fn on a single node, then
    teardown cluster. Requires an MPI environment with at least 3 cpus.
    Logs output of ipcontroller and ipengines at debug level.

    fn runs on Node 0
    ipcontroller runs on node 1
    ipengines runs on all node 2+

    Parameters
    ----------
    fn : the function to be called

    Returns
    -------
    result : the result returned by fn. Only returned by node 1, else None.

    Raises
    ------
    RuntimeError : If the ncpus in the MPI world is < 3

    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    result = None

    if size < 3:
        raise RuntimeError("At least 3 cpus needed in mpi world, "
                           "found {}".format(size))
    n_engines = size - 2
    if rank == 0:
        log.info("Cluster will have {} engines".format(n_engines))

    if rank == 1:
        log.info("Running ipcontroller")
        thread = run_ipcontroller()
        log.debug("ipcontroller ready")

    #  Everyone wait until ip controller is up
    sync_to_node(1, comm)
    log.debug("All nodes sychronised after ipcontroller startup")

    #  Start the ipengines on cpus 2 and up
    if rank >= 2:
        log.info("Starting ipengine {}".format(rank-2))
        thread = run_ipengine()
        ipengine_started = np.ones(1)
        comm.Send(ipengine_started, dest=0)
        log.debug("Engine {} has announced startup".format(rank-2))

    # Node 1 waits until all engines are up
    if rank == 0:
        num_engines_started = 0
        while num_engines_started < n_engines:
            started = np.zeros(1)
            comm.Recv(started, source=MPI.ANY_SOURCE)
            num_engines_started += 1
            log.info("Root node reports {} of {} engines ready".format(
                     num_engines_started, n_engines))

        # All engines ready
        log.info("Root node starting main command")
        result = fn()
        log.info("Root node has completed main command")

    # make everyone wait for the main task to finish on task zero
    sync_to_node(0, comm)
    if rank != 0:
        thread.terminate()
    log.info("Node exited successfully")
    return result


class ThreadWrapper:
    """
    Tiny wrapper of a thread object that contains an is_finished handle.
    Only used to call terminate and correctly kill the thread when needed
    """
    def __init__(self, thread, end_flag):
        self.thread = thread
        self.end_flag = end_flag

    def terminate(self):
        self.end_flag.set()
        self.thread.join()


def run_and_wait(command_list, output_string, timeout=None):
    """
    Run an external subcommand in own thread, which has a 'ready condition'

    This function runs command_list in a subprocess in a second thread, then
    waits until the output_string is found in the output of that subprocess. It
    then returns a wrapper to the thread in which the command continues to run.
    This thread wrapper can be subsequently used to terminate the subprocess.

    Parameters
    ----------
    command_list : list of strings to pass to the subprocess command
    output_string : the string which subprocess outputs to indicate it's ready
    timeout: optional integer number of seconds to wait for ready signal
             (default infinity)

    Returns
    -------
    thread : ThreadWrapper object that will correctly terminate the subprocess
             (and the thread) when requested
    Raises
    ------
    RuntimeError : If the ready string is not found within the timeout
    """
    is_ready = threading.Event()
    is_finished = threading.Event()
    subproc_thread = threading.Thread(target=run_with_check,
                                      args=(command_list, output_string,
                                            is_ready, is_finished, timeout))
    subproc_thread.start()
    log.debug("Started subproc, waiting to be ready")
    is_ready.wait()
    log.debug("Subproc is ready")
    result = ThreadWrapper(subproc_thread, is_finished)
    return result


def run_with_check(command_list, output_string,
                   is_ready, is_finished, timeout=None):
    """
    Run an external subcommand, which has a 'ready condition'

    This function runs command_list in a subprocess, then waits until
    the output_string is found in the output of that subprocess. It then
    returns a wrapper to the thread in which the command continues to run.
    This thread wrapper can be subsequently used to terminate the subprocess.

    Parameters
    ----------
    command_list : list of strings to pass to the subprocess command
    output_string : the string which subprocess outputs to indicate it's ready
    is_ready : threading event to signal when subprocess is ready
    is_finish : threading event on which to listen for shutdown signal
    timeout: optional integer number of seconds to wait for ready signal
             (default infinity)

    Raises
    ------
    RuntimeError : If the ready string is not found within the timeout
    """
    # note bufsize=0 is critical, lines sometimes get missed otherwise
    p = subprocess.Popen(command_list, stdin=None,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         shell=False, bufsize=0)

    poll_obj = select.poll()
    poll_obj.register(p.stdout, select.POLLIN)

    start_time = time.time()
    ready = False
    timedout = False
    done = False
    while not timedout and not done:
        if poll_obj.poll(1):
            line = p.stdout.readline().decode()
            log.debug(line)
            if not ready:
                if output_string in line:
                    log.debug("Command {} now has ready status".format(
                        command_list))
                    is_ready.set()
                    ready = True

        if not ready and timeout and time.time() - start_time > timeout:
            timedout = True

        # check we're not done
        if is_finished.is_set():
            done = True

    p.terminate()
    if timedout:
        raise RuntimeError("Process timed out without string match")
    else:
        log.debug("subprocess successfully terminated")


def run_ipcontroller():
    """
    Run ipcontroller in separate thread, return when ready with handle to
    thread for subsequent termination

    Returns
    -------
    proc : ThreadWrapper to terminate process when required

    """
    cmd_list = ["ipcontroller"]
    connected_string = "connected"
    proc = run_and_wait(cmd_list, connected_string)
    return proc


def run_ipengine():
    """
    Run ipengine in separate thread, return when ready with handle to
    thread for subsequent termination

    Returns
    -------
    proc : ThreadWrapper to terminate process when required

    """
    cmd_list = ["ipengine"]
    connected_string = "Completed registration with id"
    proc = run_and_wait(cmd_list, connected_string)
    return proc


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
        time.sleep(5)

    call_with_ipympi(fn)

if __name__ == "__main__":
    main()
