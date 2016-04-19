import logging
from os import path
from subprocess import check_call, CalledProcessError

log = logging.getLogger(__name__)


#
# CLI running and caching functions
#

class PipeLineFailure(Exception):
    pass


def try_run_checkfile(cmd, checkfile, premsg=None):
    # TODO make this a proper memoize function?

    if not path.exists(checkfile):
        if premsg is not None:
            log.info(premsg)
        try_run(cmd)
        return True

    return False


def try_run(cmd):

    try:
        check_call(cmd)
    except CalledProcessError:
        log.info("\n--------------------\n")
        raise
