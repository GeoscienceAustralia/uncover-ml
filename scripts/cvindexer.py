import sys
import json
import logging
import click as cl

from uncoverml import geom

log = logging.getLogger(__name__)


@cl.command()
@cl.option('--folds', required=False, help="Number of folds for cross"
           "validation", type=int)
@cl.option('--grid', is_flag=True, help="Using grid pointspec input data")
@cl.option('--verbose', is_flag=True, help="Log verbose output")
@cl.argument('pointspec', type=cl.Path(exists=False), required=True)
def main(pointspec, grid, verbose=False, folds=5):

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

    # Establish size of dataset
    N = pointobj.coords.shape[0]
    print(N)

    # Make fold indices associated with the coordinates/grid
    #  see uncoverml.validation

    # Write out an HDF5

if __name__ == "__main__":
    main()
