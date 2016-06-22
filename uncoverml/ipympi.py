import logging
import select
import threading
import time
import subprocess
import ipyparallel as ipp

from mpi4py import MPI

# Globals from MPI and logging
log = logging.getLogger(__name__)


def barrier(comm):
    req = comm.Ibarrier()
    while not req:
        time.sleep(1)


def waitfor_n_engines(n):
    client = ipp.Client()
    while len(client) < n:
        time.sleep(0.1)


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
    barrier()
    # sync_to_node(1, comm)
    log.debug("All nodes sychronised after ipcontroller startup")

    #  Start the ipengines on cpus 2 and up
    if rank >= 2:
        log.info("Starting ipengine {}".format(rank - 2))
        thread = run_ipengine()

    # Node 1 waits until all engines are up
    try:
        if rank == 0:
            waitfor_n_engines(n_engines)
            log.info("Root node reports {} engines ready".format(n_engines))
            log.info("Root node starting main command")
            result = fn()
            log.info("Root node has completed main command")
    finally:
        # make everyone wait for the main task to finish on task zero
        barrier()
        # sync_to_node(0, comm)
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
    log.debug("Started subprocess, waiting to be ready")
    ready_flagged = is_ready.wait(timeout=timeout)
    if not ready_flagged:
        is_finished.set()
        subproc_thread.join()
        raise RuntimeError("Timout reached before subprocess ready")
    log.debug("Subprocess is ready")
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

    """
    # note bufsize=0 is critical, lines sometimes get missed otherwise
    p = subprocess.Popen(command_list, stdin=None, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=False, bufsize=0)

    poll_obj = select.poll()
    poll_obj.register(p.stdout, select.POLLIN)

    ready = False
    while not is_finished.is_set():
        if poll_obj.poll(1):
            line = p.stdout.readline().decode()
            if line != "":
                log.debug(line)
            if not ready:
                if output_string in line:
                    log.debug("Command {} now has ready status".format(
                        command_list))
                    is_ready.set()
                    ready = True
        time.sleep(0.01)

    p.terminate()
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
