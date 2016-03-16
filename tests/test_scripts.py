import os
import tables
import json
import numpy as np

from uncoverml.scripts.maketargets import main as maketargets
from uncoverml import geom


def test_make_targets(make_shp_gtiff):

    fshp, _ = make_shp_gtiff
    field = "lon"

    maketargets.callback(shapefile=fshp, fieldname=field, outfile=None,
                         quiet=False)

    fjson = os.path.splitext(fshp)[0] + "_" + field + ".json"
    fhdf5 = os.path.splitext(fshp)[0] + "_" + field + ".hdf5"

    assert os.path.exists(fjson)
    assert os.path.exists(fhdf5)

    with open(fjson, 'r') as f:
        jdict = json.load(f)
        lspec = geom.ListPointSpec._from_json_dict(jdict)

    with tables.open_file(fhdf5, mode='r') as f:
        lon = np.array([i for i in f.root.lon])

    assert np.allclose(lon, lspec.xcoords)
