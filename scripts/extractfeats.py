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
import time
import pyprind

log = logging.getLogger(__name__)

def check_is_subset(geotiff, pointspec):
    with rasterio.open(geotiff) as raster:
        x_range, y_range = geom.bounding_box(raster)
        res = (raster.width, raster.height)
        tifgrid = geom.GridPointSpec(x_range, y_range, res)
    # Check we have valid bounding boxes
    if not tifgrid.contains(pspec):
        log.fatal("The input geotiff does not contain the pointspec data!")
        sys.exit(-1)

def print_celery_progress(async_results, title):
    total_jobs = len(async_results)
    bar = pyprind.ProgBar(total_jobs, width=60, title=title)
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


#TODO make these defaults come from uncoverml.defaults
@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.option('--patchsize', type=int, default=0, help="window size of patches")
@cl.option('--splits', type=int, default=3,
           help="Per-axis splits for chunking. Total chunks is square of this")
@cl.option('--redisdb', type=int, default=0)
@cl.option('--redishost', type=str, default='localhost')
@cl.option('--redisport', type=int, default=6379)
@cl.option('--standalone', is_flag=True, default=False)
@cl.argument('pointspec', type=cl.Path(exists=True), required=True)
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
@cl.argument('outfile', type=cl.Path(exists=False), required=True)
def main(geotiff, pointspec, outfile, patchsize, splits, quiet, 
         redisdb, redishost, redisport, standalone):

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # initialise celery
    if not standalone:
        from uncoverml import celerybase
        celerybase.configure(redishost, redisport, redisdb)
    
    # Read in the poinspec file and create the relevant object
    with open(pointspec, 'r') as f:
        jdict = json.load(f)
    pspec = geom.unserialise(jdict)

    # TODO: figure out if we need to rescale geotiff accoding to pspec if grid?
    # This will also affect the pstride value, though for the moment keep as 1
    # at the moment I'm totally ignoring resizing

    # Define the transform function to build the features
    transform = feat.transform
    
    # Build the chunk indices for creating jobs
    chunk_indices = [(x, y) for x in range(splits) for y in range(splits)]

    # Send off the jobs
    progress_title = "Processing Image Chunks"
    if not standalone:
        async_results = []
        for x, y in chunk_indices:
            r = feat.process_window.delay(x,y, splits, geotiff, pspec, 
                    patchsize, transform, outfile)
            async_results.append(r)
        print_celery_progress(async_results, progress_title)
    else:
        bar = pyprind.ProgBar(len(chunk_indices), width=60, title=progress_title)
        for x, y in chunk_indices:
            r = feat.process_window(x,y, splits, geotiff, pspec, 
                    patchsize, transform, outfile)
            bar.update(force_flush=True)

    sys.exit(0)

