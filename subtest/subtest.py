import logging
import select
import threading
import time
import numpy as np
import subprocess
from mpi4py import MPI

log = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def run_with_ipyparallel(command_list):
    if size < 3:
        raise RuntimeError("At least 3 cpus needed in mpi world, "
                           "found {}".format(size))
    n_engines = size - 2
    log.info("Cluster will have {} engines".format(n_engines))

    if rank == 1:
        log.info("CPU 1 running ipcontroller")
        thread = run_ipcontroller()

    #  Everyone wait until ip controller is up
    log.info("CPU 1 knows ipcontroller started, broadcasting")
    ipc_started = np.zeros(1)
    ipc_started = comm.bcast(ipc_started, root=1)
    log.info("All nodes sychronised after ipcontroller startup")

    #  Start the ipengines on cpus 2 and up
    if rank >= 2:
        log.info("Starting ipengine {} on node {}".format(rank-2, rank))
        thread = run_ipengine()
        ipengine_started = np.ones(1)
        comm.Send(ipengine_started, dest=0)
        log.info("Engine {} has announced startup".format(rank-2))

    if rank == 0:
        num_engines_started = 0
        while num_engines_started < n_engines:
            started = np.zeros(1)
            comm.Recv(started, source=MPI.ANY_SOURCE)
            num_engines_started += 1
            log.info("Root node reports {} of {} engines ready".format(
                     num_engines_started, n_engines))

        log.info("Root node starting main command")
        subprocess.run(command_list)
        log.info("Root node has completed main command")

    # make everyone wait for the main task to finish on task zero
    if rank != 0:
        log.info("Waiting for shutdown signal")
    task_done = np.ones(1)
    task_done = comm.bcast(task_done, root=0)
    if rank != 0:
        log.info("Shutting down cluster")
        thread.terminate()
    log.info("Node finished without error")


class ThreadWrapper:
    def __init__(self, thread, end_flag):
        self.thread = thread
        self.end_flag = end_flag

    def terminate(self):
        self.end_flag.set()
        self.thread.join()


def run_and_wait(command_list, output_string, timeout=None):
    is_ready = threading.Event()
    is_finished = threading.Event()
    subproc_thread = threading.Thread(target=run_with_check,
                                      args=(command_list, output_string,
                                            is_ready, is_finished, timeout))
    subproc_thread.start()
    logging.info("node {} started subproc, waiting to be ready".format(rank))
    is_ready.wait()
    logging.info("node {} subproc is ready".format(rank))
    result = ThreadWrapper(subproc_thread, is_finished)
    return result


def run_with_check(command_list, output_string,
                   is_ready, is_finished, timeout=None):
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
                    log.info("Command {} now has ready status".format(
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
        log.info("process on node {} successfully terminated "
                 "after finish flag raised".format(rank))


def run_ipcontroller():
    cmd_list = ["ipcontroller"]
    connected_string = "connected"
    proc = run_and_wait(cmd_list, connected_string)
    return proc


def run_ipengine():
    cmd_list = ["ipengine"]
    connected_string = "Completed registration with id"
    proc = run_and_wait(cmd_list, connected_string)
    return proc


def main():
    logging.basicConfig(level=logging.INFO,
                        format='node{} %(threadName)s %(relativeCreated)6d'
                        ' %(levelname)s: %(message)s'.format(rank))
    # logging.basicConfig(filename='node{}.log'.format(rank),
    #                     level=logging.INFO)

    run_with_ipyparallel(["python", "wait.py"])

if __name__ == "__main__":
    main()
