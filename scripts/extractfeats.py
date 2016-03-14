import logging
import sys
import os.path
import click as cl
import numpy as np
import rasterio
import json
import uncoverml.geom as geom
import uncoverml.patch as patch
import uncoverml.feature as feat
from celery import Celery
import time
import pyprind

log = logging.getLogger(__name__)

@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.option('--patchsize', type=int, default=0, help="window size of patches")
@cl.option('--chunks', type=int, default=10,
           help="Number of chunks to divide work into")
@cl.option('--redisdb', type=int, default=0)
@cl.option('--redishost', type=str, default='localhost')
@cl.option('--redisport', type=int, default=6379)
@cl.argument('pointspec', type=cl.Path(exists=True), required=True)
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
@cl.argument('outfile', type=cl.Path(exists=False), required=True)
def main(geotiff, pointspec, outfile, patchsize, chunks, quiet, 
         redisdb, redishost, redisport):

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # initialise celery
    celery = Celery('uncoverml.tasks')
    celery_redis='redis://{}:{}/{}'.format(redishost, redisport, redisdb)
    celery.conf.BROKER_URL = celery_redis
    celery.conf.CELERY_RESULT_BACKEND = celery_redis


    # Read in the poinspec file and create the relevant object
    with open(pointspec, 'r') as f:
        jdict = json.load(f)

    pspec = geom.unserialise(jdict)

    # Get geotiff properties
    with rasterio.open(geotiff) as raster:
        x_range, y_range = geom.bounding_box(raster)
        res = (raster.width, raster.height)
        tifgrid = geom.GridPointSpec(x_range, y_range, res)

    # Check we have valid bounding boxes
    if not tifgrid.contains(pspec):
        log.fatal("The input geotiff does not contain the pointspec data!")
        sys.exit(-1)

    # TODO: figure out if we need to rescale geotiff accoding to pspec if grid?
    # This will also affect the pstride value, though for the moment keep as 1
    # at the moment I'm totally ignoring resizing
    pstride = 1

    slices = patch.image_windows(tifgrid.resolution, chunks, 
                                 patchsize, pstride)
    
    async_results = [celery.send_task("process_window", [geotiff, i, s,
        pspec, patchsize, feat.transform, outfile] ) for i, s in enumerate(slices)]


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

