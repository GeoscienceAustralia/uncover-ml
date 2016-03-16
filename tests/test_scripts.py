import os
import tables
import json
import numpy as np
import shapefile
import rasterio
from affine import Affine

from uncoverml import geom
from uncoverml.scripts.maketargets import main as maketargets
from uncoverml.scripts.pointspec import main as pointspec


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


def test_pointspec(make_shp_gtiff):

    fshp, ftif = make_shp_gtiff
    fshp_json = os.path.splitext(fshp)[0] + "_" + ".json"
    ftif_json = os.path.splitext(ftif)[0] + "_" + ".json"

    # Test shapefile reading
    pointspec.callback(fshp_json, pointlist=fshp, resolution=None,
                       bbox=None, geotiff=None, quiet=False)

    with open(fshp_json, 'r') as f:
        jdict = json.load(f)
        lspec = geom.ListPointSpec._from_json_dict(jdict)

    f = shapefile.Reader(fshp)
    lonlat = np.array([p.points[0] for p in f.shapes()])

    assert np.allclose(lonlat, lspec.coords)

    # Test geotiff reading
    pointspec.callback(ftif_json, geotiff=ftif, resolution=None, bbox=None,
                       pointlist=None, quiet=False)

    with open(ftif_json, 'r') as f:
        jdict = json.load(f)
        gspec = geom.GridPointSpec._from_json_dict(jdict)

    with rasterio.open(ftif, 'r') as f:
        A = f.affine * Affine.translation(0.5, 0.5)
        bbox = np.array([f.bounds[slice(0, None, 2)],
                         f.bounds[slice(1, None, 2)]])
        bbox[1][0] -= f.res[1]
        bbox[0][1] += f.res[0]
        res = (f.width, f.height)

    assert np.allclose(A, gspec.A)
    assert np.allclose(bbox, gspec.bbox)
    assert np.allclose(res, gspec.resolution)

    # Test bounding box
    bstring = "10:20,-30:-20"
    rstring = "10x10"
    pointspec.callback(ftif_json, bbox=bstring, resolution=rstring,
                       geotiff=None, pointlist=None, quiet=False)

    with open(ftif_json, 'r') as f:
        jdict = json.load(f)
        gspec = geom.GridPointSpec._from_json_dict(jdict)

    assert np.allclose([[10, 20], [-30, -20]], gspec.bbox)
    assert np.allclose([10, 10], gspec.resolution)
