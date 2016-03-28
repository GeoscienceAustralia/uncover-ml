"""
Compose multiple image features into a single feature vector.

.. program-output:: composefeats --help
"""
import logging
import sys
import os.path
import click as cl
import numpy as np
import json
import time
import pyprind
import uncoverml.defaults as df
from uncoverml import celerybase
from uncoverml import geoio

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
    files_ok = geoio.file_indices_okay(files)
    if not files_ok:
        sys.exit(-1)
    
        
    # build the images
    filename_chunks = geoio.files_by_chunk(files)
    nchunks = len(filename_chunks)
    images = [[geoio.Image(f, i, nchunks) for f in filename_chunks[i]]
              for i in range(nchunks)]
    
    # Define the transform function to build the features
    

    celerybase.map_over(feat.features_from_image, image_chunks, standalone,
                        name=name, transform=transform, patchsize=patchsize,
                        output_dir=outputdir, targets=targets)

    sys.exit(0)

