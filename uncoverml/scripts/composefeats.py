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

@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.option('--redisdb', type=int, default=0)
@cl.option('--redishost', type=str, default='localhost')
@cl.option('--redisport', type=int, default=6379)
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(quiet, redisdb, redishost, redisport, files):

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # initialise celery
    celery_redis='redis://{}:{}/{}'.format(redishost, redisport, redisdb)
    celery.conf.BROKER_URL = celery_redis
    celery.conf.CELERY_RESULT_BACKEND = celery_redis

    # check sanity of files
    basenames = [os.path.splitext(os.path.basename(k))[0] for k in files]
    base_and_idx = [k.rsplit('_', maxsplit=1) for k in basenames]
    bases = set([k[0] for k in base_and_idx])
    log.info("Input file sets: {}".format(set(bases)))
    
    # check every base has the right indices
    base_ids = {k:set([int(j[1]) for j in base_and_idx if j[0] == k]) for k in bases}
    num_ids = np.amax(np.array([max(k) for j,k in base_ids.items()])) + 1
    true_set = set(range(num_ids))
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
    if files_ok is False:
        sys.exit(-1)

    log.info("Verified feature {} has {} files".format(b, len(nums)))
    
    total_jobs = len(async_results)
    bar = pyprind.ProgBar(total_jobs, width=60, title="Processing Image Chunks")
    last_jobs_done = 0
    jobs_done = 0
    while jobs_done < total_jobs:
        jobs_done = 0
        for r in async_results:
            jobs_done += int(r.ready())
        if jobs_done > last_jobs_done:
            bar.update(jobs_done - last_jobs_done, force_flush=True)
            last_jobs_done = jobs_done
        time.sleep(0.1)

    sys.exit(0)

