"""
Make output target files.

.. program-output:: maketargets --help
"""
import os
import sys
import logging
import click as cl
import numpy as np

from uncoverml import geoio
from uncoverml.validation import output_targets

log = logging.getLogger(__name__)


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.option('--outfile', type=cl.Path(exists=False), default=None,
           help="File name (minus extension) to give to the output files")
@cl.argument('shapefile', type=cl.Path(exists=True), required=True)
@cl.argument('fieldname', type=str, required=True)
def main(shapefile, fieldname, outfile, quiet):
    """
    Turn a shapefile of target variables into an HDF5 file. This file can
    subsequently be used by cross validation and machine learning routines.

    The output hdf5 will have three arrays, "targets", "Longitude" and
    "Latitude".
    """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Extract data from shapefile
    try:
        lonlats = geoio.points_from_shp(shapefile)
        vals = geoio.values_from_shp(shapefile, fieldname)
    except Exception as e:
        log.fatal("Error parsing shapefile: {}".format(e))
        sys.exit(-1)

    # Get output file name
    if outfile is None:
        outfile = os.path.splitext(shapefile)[0] + "_" + fieldname
    else:
        # Strip output file ext always
        outfile = os.path.splitext(outfile)[0]

    # Make hdf5 array
    output_targets(vals, lonlats, outfile + ".hdf5")
