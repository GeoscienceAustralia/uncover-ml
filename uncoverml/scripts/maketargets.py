"""
Make output target file with cross-validation indices.

.. program-output:: maketargets --help
"""
import os
import sys
import logging

import click as cl
import click_log as cl_log
import numpy as np
from mpi4py import MPI

from uncoverml import geoio
from uncoverml.validation import split_cfold

log = logging.getLogger(__name__)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--outfile', type=cl.Path(exists=False), default=None,
           help="File name (minus extension) to give to the output files")
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

    # MPI globals
    comm = MPI.COMM_WORLD
    chunk_index = comm.Get_rank()
    # This runs on the root node only
    if chunk_index != 0:
        return

    # Extract data from shapefile
    try:
        lonlat, vals = geoio.load_shapefile(shapefile, fieldname)
    except Exception as e:
        log.fatal("Error parsing shapefile: {}".format(e))
        sys.exit(-1)

    # Get output file name
    if outfile is None:
        outfile = os.path.splitext(shapefile)[0] + "_" + fieldname
    else:
        # Strip output file ext always
        outfile = os.path.splitext(outfile)[0]

    # Make fold indices associated with the coordinates/grid
    N = len(lonlat)
    _, cvassigns = split_cfold(N, folds, seed)

    # Get ascending order of targets by lat then lon
    ordind = np.lexsort(lonlat.T)

    # Make field dict for writing to HDF5
    fielddict = {
        'targets': vals,
        'Longitude': lonlat[:, 0],
        'Latitude': lonlat[:, 1],
        'FoldIndices': cvassigns,
        'targets_sorted': vals[ordind],
        'Longitude_sorted': lonlat[:, 0][ordind],
        'Latitude_sorted': lonlat[:, 1][ordind],
        'FoldIndices_sorted': cvassigns[ordind]
    }

    # Writeout
    geoio.points_to_hdf(outfile + ".hdf5", fielddict)

    # Make hdf5 array
    # output_targets(vals, lonlat, cvassigns, outfile + ".hdf5")

    # Write out an HDF5
    # output_cvindex(cvassigns, lonlat, outfile)
