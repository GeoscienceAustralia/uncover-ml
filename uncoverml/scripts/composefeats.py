import logging
import sys
import os.path
import click as cl
import numpy as np
import json
from uncoverml.celerybase import celery
import time
import pyprind

log = logging.getLogger(__name__)



@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", 
           default=df.quiet_logging)
@cl.option('--redisdb', type=int, default=df.redis_db)
@cl.option('--redishost', type=str, default=df.redis_address)
@cl.option('--redisport', type=int, default=df.redis_port)
@cl.option('--standalone', is_flag=True, default=df.standalone)
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('name', type=str, required=True) 
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(files, name, redisdb, redishost, redisport, 
         standalone, quiet, outputdir):
    """ TODO
    """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # initialise celery
    celerybase.configure(redishost, redisport, redisdb, standalone)

    # verify the files are all present
    files_ok = file_indices_okay(files)
    if not files_ok:
        sys.exit(-1)
    
        
    # build the images
    image_chunks = [geoio.Image(geotiff, i, chunks) for i in range(chunks)]

    # Define the transform function to build the features
    transform = feat.transform

    celerybase.map_over(feat.features_from_image, image_chunks, standalone,
                        name=name, transform=transform, patchsize=patchsize,
                        output_dir=outputdir, targets=targets)

    sys.exit(0)

