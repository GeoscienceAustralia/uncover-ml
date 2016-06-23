"""
Create a cross validation fold index.

.. program-output:: cvindexer --help
"""
import os
import sys
import logging
import click as cl
import click_log as cl_log
from uncoverml import validation

log = logging.getLogger(__name__)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--folds', required=False, help="Number of folds for cross"
           "validation", type=int, default=5)
@cl.argument('targetfile', type=cl.Path(exists=True), required=True)
@cl.argument('outfile', type=cl.Path(exists=False), required=True)
def main(targetfile, outfile, folds):
    """
    Create a cross validation fold index file from a shapefile or HDF5 of
    appropriate formats. This outputs an HDF5 file with "Latitude",
    "Longitude", and "FoldIndices" arrays.
    """

    # Try to read in shapefile or hdf5
    ext = os.path.splitext(targetfile)[-1]
    if ext == ".hdf5":
        lonlat, _ = validation.input_targets(targetfile)
    else:
        log.fatal("Invalid file type, {}, need *.hdf5".format(ext))
        sys.exit(-1)

    # Make fold indices associated with the coordinates/grid
    N = len(lonlat)
    _, cvassigns = validation.split_cfold(N, folds)

    # make sure outfile has an hdf extension
    outsplit = os.path.splitext(outfile)
    outfile = outsplit[0] + ".hdf5"

    # Write out an HDF5
    validation.output_cvindex(cvassigns, lonlat, outfile)
