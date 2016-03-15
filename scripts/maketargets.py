import os
import sys
import json
import logging
import tables
import click as cl

from uncoverml import geom

log = logging.getLogger(__name__)


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.option('--outfile', type=cl.Path(exists=False), default=None,
           help="File name (minus extension) to give to the output files")
@cl.argument('shapefile', type=cl.Path(exists=True), required=True)
@cl.argument('fieldname', type=str, required=True)
def main(shapefile, fieldname, outfile, quiet):
    """
    Turn a shapefile of target variables into a point-spec file and HDF5 file.
    These files can subsequently be used by cross validation and machine
    learning routines.
    """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Extract data from shapefile
    try:
        lonlats = geom.points_from_shp(shapefile)
        vals = geom.values_from_shp(shapefile, fieldname)
    except Exception as e:
        log.fatal("Error parsing shapefile: {}".format(e))
        sys.exit(-1)

    # Get output file name
    if outfile is None:
        outfile = os.path.splitext(shapefile)[0] + "_" + fieldname

    # Make pointspec json
    lspec = (geom.ListPointSpec(lonlats))._to_json_dict()
    with open(outfile + ".json", 'w') as f:
        json.dump(lspec, f)

    # Make hdf5 array
    with tables.open_file(outfile + ".hdf5", 'w') as f:
        f.create_array("/", fieldname, obj=vals)
