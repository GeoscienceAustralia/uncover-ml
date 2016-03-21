"""
Create a cross validation fold index.

.. program-output:: cvindexer --help
"""
import os
import sys
import logging
import tables
import click as cl
import numpy as np

from uncoverml import geoio, validation

log = logging.getLogger(__name__)


@cl.command()
@cl.option('--folds', required=False, help="Number of folds for cross"
           "validation", type=int, default=5)
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.argument('targetfile', type=cl.Path(exists=True), required=True)
@cl.argument('outfile', type=cl.Path(exists=False), required=True)
def main(targetfile, outfile, folds, quiet):
    """
    Create a cross validation fold index file from a shapefile or HDF5 of
    appropriate formats. This outputs an HDF5 file with "Latitude",
    "Longitude", and "FoldIndices arrays.
    """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Try to read in shapefile or hdf5
    ext = os.path.splitext(targetfile)[-1]
    if ext == ".shp":
        lonlat = geoio.points_from_shp(targetfile)
    elif (ext == ".hdf5") or (ext == ".hdf"):
        with tables.open_file(targetfile, mode='r') as f:
            Longitude = [l for l in f.root.Longitude]
            Latitude = [l for l in f.root.Latitude]
        lonlat = np.array([Longitude, Latitude]).T
    else:
        log.fatal("Invalid file type, {}, need *.shp or *.hdf(5)".format(ext))
        sys.exit(-1)

    # Make fold indices associated with the coordinates/grid
    N = len(lonlat)
    _, cvassigns = validation.split_cfold(N, folds)

    # make sure outfile has an hdf extension
    outsplit = os.path.splitext(outfile)
    outfile = outsplit[0] + ".hdf5" if outsplit[1] != ".hdf5" else outfile

    # Write out an HDF5
    with tables.open_file(outfile, 'w') as f:
        f.create_array("/", "Longitude", obj=lonlat[:, 0])
        f.create_array("/", "Latitude", obj=lonlat[:, 1])
        f.create_array("/", "FoldIndices", obj=cvassigns)
