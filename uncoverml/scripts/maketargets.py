"""
Make output target files.

.. program-output:: maketargets --help
"""
import os
import sys
import logging
import click as cl
import click_log as cl_log

from uncoverml import geoio
from uncoverml.validation import output_targets

log = logging.getLogger(__name__)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--outfile', type=cl.Path(exists=False), default=None,
           help="File name (minus extension) to give to the output files")
@cl.argument('shapefile', type=cl.Path(exists=True), required=True)
@cl.argument('fieldname', type=str, required=True)
def main(shapefile, fieldname, outfile):
    """
    Turn a shapefile of target variables into an HDF5 file. This file can
    subsequently be used by cross validation and machine learning routines.

    The output hdf5 will have three arrays, "targets", "Longitude" and
    "Latitude".
    """

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
