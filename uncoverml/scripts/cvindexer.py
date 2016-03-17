import os
import sys
import json
import logging
import tables
import click as cl

from uncoverml import geom, validation

log = logging.getLogger(__name__)


@cl.command()
@cl.option('--folds', required=False, help="Number of folds for cross"
           "validation", type=int, default=5)
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.argument('pointspec', type=cl.Path(exists=True), required=True)
@cl.argument('outfile', type=cl.Path(exists=False), required=True)
def main(pointspec, outfile, folds, quiet):
    """
    Create a cross validation fold index file from a pointspec. This outputs an
    HDF5 file with "Latitude", "Longitude", and "FoldIndices arrays. If the
    pointspec is a grid, then "FoldIndices" is a 2D array.
    """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read in the poinspec file and create the relevant object
    with open(pointspec, 'r') as f:
        jdict = json.load(f)
    pspec = geom.unserialise(jdict)

    if isinstance(pspec, geom.GridPointSpec):
        # TODO: implement this
        log.fatal("Grid cross validation not implemented yet, sorry!")
        sys.exit(-1)

    pointobj = geom.ListPointSpec._from_json_dict(jdict)

    # Make fold indices associated with the coordinates/grid
    N = pointobj.coords.shape[0]
    _, cvassigns = validation.split_cfold(N, folds)

    # make sure outfile has an hdf extension
    outsplit = os.path.splitext(outfile)
    outfile = outsplit[0] + ".hdf5" if outsplit[1] != ".hdf5" else outfile

    # Write out an HDF5
    with tables.open_file(outfile, 'w') as f:
        f.create_array("/", "Longitude", obj=pointobj.xcoords)
        f.create_array("/", "Latitude", obj=pointobj.ycoords)
        f.create_array("/", "FoldIndices", obj=cvassigns)
