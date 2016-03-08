import logging
import sys
import click as cl
import numpy as np
import rasterio
import json
import uncoverml.geom as geom

log = logging.getLogger(__name__)


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.option('--patchsize', type=int, default=1, help="window size of patches")
@cl.option('--chunks', type=int, default=10,
           help="Number of chunks to divide work into")
@cl.argument('pointspec', type=cl.Path(exists=True), required=True)
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
@cl.argument('outfile', type=cl.Path(exists=False), required=True)
def main(geotiff, pointspec, outfile, patchsize, chunks, quiet):

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

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

    print(tifgrid.x_range, tifgrid.y_range)
    print(pspec.x_range, pspec.y_range)

    # Compute lists of pixels to calculate features on per task, i.e. chunk
    # work (should this be an input option with defaults?)
    #   ListPointSpec -- divide up list, send to workers
    #   GridPointSpec -- split work into windows, send windows to workers
    #       NOTE: These windows need to account for patch/stride overlaps!

    # Start workers (celery)

    # Report status

    # Report success or fail
