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
@cl.option('--grid', is_flag=True, help="Using grid pointspec input data")
@cl.option('--verbose', is_flag=True, help="Log verbose output", default=False)
@cl.argument('pointspec', type=cl.Path(exists=True), required=True)
@cl.argument('outfile', type=cl.Path(exists=False), required=True)
def main(pointspec, outfile, folds, grid, verbose):

    # setup logging
    if verbose is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if grid:
        # TODO: implement this
        log.fatal("Grid cross validation not implemented yet, sorry!")
        sys.exit(-1)

    # Read in the poinspec file and create the relevant object
    with open(pointspec, 'r') as f:
        jdict = json.load(f)

    pointobj = geom.ListPointSpec._from_json_dict(jdict)

    # Make fold indices associated with the coordinates/grid
    N = pointobj.coords.shape[0]
    _, cvassigns = validation.split_cfold(N, folds)

    # Write out an HDF5
    with tables.open_file(outfile, 'w') as f:
        f.create_array("/", "Longitude", obj=pointobj.xcoords)
        f.create_array("/", "Latitude", obj=pointobj.ycoords)
        f.create_array("/", "FoldIndices", obj=cvassigns)


if __name__ == "__main__":
    main()
