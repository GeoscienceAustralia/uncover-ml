import logging
import sys
import os
import click as cl
import json
import uncoverml.feature as feat
from uncoverml import io
import time
import pyprind
from uncoverml import celerybase
log = logging.getLogger(__name__)


def check_is_subset(geotiff, pointspec):
    with io.open_raster(geotiff) as raster:
        x_range, y_range = geom.bounding_box(raster)
        res = (raster.width, raster.height)
        tifgrid = geom.GridPointSpec(x_range, y_range, res)

    # Check we have valid bounding boxes
    if not tifgrid.contains(pointspec):
        log.fatal("The input geotiff does not contain the pointspec data!")
        sys.exit(-1)

# TODO make these defaults come from uncoverml.defaults
@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.option('--patchsize', type=int, default=0, help="window size of patches")
@cl.option('--chunks', type=int, default=10,
           help="Number of chunks in which to split the computation and output")
@cl.option('--redisdb', type=int, default=0)
@cl.option('--redishost', type=str, default='localhost')
@cl.option('--redisport', type=int, default=6379)
@cl.option('--standalone', is_flag=True, default=False)
@cl.option('--targets', type=cl.Path(exists=True), help="Optional shapefile "
           "for providing target points at which to evaluate feature")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
@cl.argument('name', type=str, required=True) 
def main(geotiff, name, targets, redisdb, redishost, redisport, standalone,
        chunks, patchsize, quiet, outputdir):
    """ TODO
    """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # initialise celery
    celerybase.configure(redishost, redisport, redisdb, standalone)

    #build the images
    image_chunks = [io.Image(geotiff, i, chunks) for i in range(chunks)]

    # Define the transform function to build the features
    transform = feat.transform

    celerybase.map_over(feat.features_from_image, image_chunks, standalone, 
             name=name, transform=transform, patchsize=patchsize,
             output_dir=outputdir, targets=targets)
