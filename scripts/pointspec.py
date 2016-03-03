import logging
import sys
import click as cl
import numpy as np
import rasterio
import json
import uncoverml.geom as geom

log = logging.getLogger(__name__)


@cl.command()
@cl.option('--resolution', required=False,
           help="the output resolution. "
           "Has the form --resolution <width>x<height>")
@cl.option('--bbox', required=False,
           help="a bounding box of a subset of the image."
           "takes the form --bbox <xmin>:<xmax>,<ymin>:<ymax> and if used"
           "in conjunction with --resolution,"
           " the resolution is of the subset.")
@cl.option('--pointlist', type=cl.Path(exists=True), required=False,
           help="a shapefile containing a list of positions. Doesn't "
           "work with --resolution. If used with --bbox, only those points "
           "lying inside (and on) the bbox will be used")
@cl.option('--geotiff', type=cl.Path(exists=True), required=False,
           help="a geotiff to extract the specification from. Can be used "
           "with --resolution. If given --bbox is ignored")
@cl.option('--verbose', is_flag=True, help="Log verbose output", default=False)
@cl.argument('outfile', type=cl.Path(exists=False), required=True)
def main(outfile, resolution=None, bbox=None, pointlist=None, geotiff=None,
         verbose=False):
    """
    Builds a JSON file that encodes the latitude/longitude points being used
    for an ML problem. This resulting PointSpec file is used to ensure machine
    learning outputs can be placed back into an image or onto a map.
    """
    # setup logging
    if verbose is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if resolution is None and geotiff is None and pointlist is None:
        log.fatal("You must specify either a geotiff, a resolution, or "
                  "a pointlist")
        sys.exit(-1)

    x_range = None
    y_range = None

    if resolution is not None:
        final_res = tuple([int(k) for k in resolution.split("x")])
    if bbox is not None:
        x_range, y_range = tuple([np.array([float(k) for k in j.split(":")])
                                  for j in bbox.split(",")])
    if geotiff is not None:
        with rasterio.open(geotiff) as raster:
            x_range, y_range = geom.bounding_box(raster)
            log.info("extracted bbox from raster: {} x {}"
                     .format(x_range, y_range))
            x_res, y_res = (raster.width, raster.height)
            if resolution is None:
                final_res = (x_res, y_res)

    if pointlist is not None:
        # list of points from a shapefile
        coords = geom.points_from_shp(pointlist)
        obj = geom.ListPointSpec(coords, x_range, y_range)
    else:
        obj = geom.GridPointSpec(x_range, y_range, final_res)

    # Finally, write the output file
    with open(outfile, 'w') as o:
        json.dump(obj._to_json_dict(), o)
