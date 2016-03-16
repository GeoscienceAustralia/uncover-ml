import logging
import sys
import os.path
import click as cl
import numpy as np
import rasterio
import json
from uncoverml.celerybase import celery
import time
import pyprind

log = logging.getLogger(__name__)


def file_indices_okay(filenames):
    #get just the name eg /path/to/file_0_0.hdf5 -> file_0_0
    basenames = [os.path.splitext(os.path.basename(k))[0] for k in filenames]
    # file_0_0 -> [file,0,0]
    base_and_idx = [k.rsplit('_', maxsplit=2) for k in basenames]
    bases = set([k[0] for k in base_and_idx])
    log.info("Input file sets: {}".format(set(bases)))
    
    # check every base has the right indices
    # "[[file,0,0], [file,0,1]] -> {file:[(0,0),(0,1)]}
    base_ids = {k:set([(int(j[1]), int(j[2])) 
                       for j in base_and_idx if j[0] == k]) for k in bases}

    # determine the 'correct' number of indices (highest index we see)
    num_ids = np.amax(np.array([max(k) for j,k in base_ids.items()])) + 1
    true_set = set([(i,j) for i in range(num_ids) for j in range(num_ids)])
    files_ok = True
    for b, nums in base_ids.items():
        if not nums == true_set:
            files_ok = False
            log.fatal("feature {} has wrong files. ".format(b))
            missing = true_set.difference(nums)
            if len(missing) > 0:
                log.fatal("Missing Index: {}".format(missing))
            extra = nums.difference(true_set)
            if len(extra) > 0:
                log.fatal("Extra Index: {}".format(extra))
    return files_ok

#TODO make these defaults come from uncoverml.defaults
@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.option('--redisdb', type=int, default=0)
@cl.option('--redishost', type=str, default='localhost')
@cl.option('--redisport', type=int, default=6379)
@cl.option('--standalone', is_flag=True, default=False)
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
@cl.argument('outfile', type=cl.Path(exists=False), required=True)
def main(quiet, redisdb, redishost, redisport, standalone, files, outfile):

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # initialise celery
    if not standalone:
        from uncoverml import celerybase
        celerybase.configure(redishost, redisport, redisdb)
    
    files_ok = file_indices_okay(files)

    if not files_ok:
        sys.exit(-1)

    sys.exit(0)

