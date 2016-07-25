"""
Make output target file with cross-validation indices.

.. program-output:: maketargets --help
"""
import os
import sys
import logging

import click as cl
import click_log as cl_log

from uncoverml import geoio
from uncoverml import mpiops
from uncoverml import pipeline

log = logging.getLogger(__name__)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--outfile', type=cl.Path(exists=False), default=None,
           help="File name to give to the output files")
@cl.option('--folds', required=False, help="Number of folds for cross"
           "validation", type=int, default=5)
@cl.option('--seed', type=int, default=None, help="Integer random seed")
@cl.argument('shapefile', type=cl.Path(exists=True), required=True)
@cl.argument('fieldname', type=str, required=True)
def main(shapefile, fieldname, outfile, folds, seed):
    """
    Turn a shapefile of target variables into an HDF5 file and create a cross
    validation fold index.

    The output hdf5 will have the following arrays:

    - targets

    - Longitude

    - Latitude

    - FoldIndices

    - targets_sorted

    - FoldIndices_sorted

    - Latitude_sorted

    - Longitude_sorted

    """

    # This only runs on one chunk
    if mpiops.chunk_index != 0:
        return

    # Filenames
    shape_infile = os.path.abspath(shapefile)
    if outfile is None:
        outfile = os.path.splitext(shapefile)[0] + "_" + fieldname + ".hdf5"

    # Extract data from shapefile
    try:
        lonlat, vals = geoio.load_shapefile(shape_infile, fieldname)
    except Exception as e:
        log.fatal("Error parsing shapefile: {}".format(e))
        sys.exit(-1)

    # Create the targets
    targets = pipeline.CrossValTargets(lonlat, vals, folds, seed)

    # Write
    geoio.write_targets(targets, outfile)
