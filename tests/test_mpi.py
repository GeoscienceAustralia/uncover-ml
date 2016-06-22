import subprocess
from uncoverml import ipympi


def test_run_and_wait():
    thd = ipympi.run_and_wait(["python", "ready.py"], "ready")
    thd.terminate()
    assert True


def test_run_and_wait_timeout():
    raised = False
    try:
        thd = ipympi.run_and_wait(["python", "ready.py"], "ready", timeout=0.1)
        thd.terminate()
    except:
        raised = True
    assert raised


def test_run_ipcontroller():
    proc = ipympi.run_ipcontroller()
    proc.terminate()
    assert True


def test_run_ipengine():
    c = ipympi.run_ipcontroller()
    e = ipympi.run_ipengine()
    c.terminate()
    e.terminate()
    assert True


def test_run_nompi():
    c = ipympi.run_ipcontroller()
    e1 = ipympi.run_ipengine()
    e2 = ipympi.run_ipengine()

    ipympi.waitfor_n_engines(2)

    c.terminate()
    e1.terminate()
    e2.terminate()
    assert True


def test_withmpi():
    retcode = subprocess.call(["mpirun", "-n", "4",
                               "python", "mpihelloworld.py"])
    assert retcode == 0
