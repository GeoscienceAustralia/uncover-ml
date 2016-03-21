import os
import tables
import json
import numpy as np
import shapefile
import rasterio
import subprocess
from affine import Affine

from uncoverml import geom, patch, io
from uncoverml.scripts.maketargets import main as maketargets
from uncoverml.scripts.pointspec import main as pointspec
from uncoverml.scripts.cvindexer import main as cvindexer
from uncoverml.scripts.extractfeats import main as extractfeats


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
    fshp_json = os.path.splitext(fshp)[0] + ".json"
    ftif_json = os.path.splitext(ftif)[0] + ".json"

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


def test_cvindexer(make_shp_gtiff):

    fshp, ftif = make_shp_gtiff
    fshp_json = os.path.splitext(fshp)[0] + ".json"
    fshp_hdf5 = os.path.splitext(fshp)[0] + ".hdf5"

    # Make pointspec
    folds = 6
    pointspec.callback(fshp_json, pointlist=fshp, resolution=None, bbox=None,
                       geotiff=None, quiet=False)

    # Make crossval
    cvindexer.callback(pointspec=fshp_json, outfile=fshp_hdf5, folds=6,
                       quiet=True)

    # Read in resultant HDF5
    with tables.open_file(fshp_hdf5, mode='r') as f:
        hdfcoords = np.array([(x, y) for x, y in zip(f.root.Longitude,
                                                     f.root.Latitude)])
        finds = np.array([i for i in f.root.FoldIndices])

    # Validate order is consistent with shapefile
    f = shapefile.Reader(fshp)
    shpcoords = np.array([p.points[0] for p in f.shapes()])

    assert np.allclose(shpcoords, hdfcoords)

    # Test we have the right number of folds
    assert finds.min() == 0
    assert finds.max() == (folds - 1)


def test_extractfeats(make_shp_gtiff):

    _, ftif = make_shp_gtiff
    split = 2

    # Make pointspec
    ftif_json = os.path.splitext(ftif)[0] + ".json"
    pointspec.callback(ftif_json, geotiff=ftif, resolution=None, bbox=None,
                       pointlist=None, quiet=False)

    # Extract features from gtiff
    ffeats = os.path.splitext(ftif)[0]
    extractfeats.callback(geotiff=ftif, pointspec=ftif_json, outfile=ffeats,
                          patchsize=0, splits=split, quiet=True, redisdb=0,
                          redishost='localhost', redisport=6379,
                          standalone=True)

    # Now compare extracted features to geotiff
    with rasterio.open(ftif, 'r') as f:
        I = np.transpose(f.read(), [2, 1, 0])

    dfeats = {(x, y): I[x, y, :]
              for x in range(I.shape[0])
              for y in range(I.shape[1])
              }

    efeats = []
    ecentres = []
    for i in range(split):
        for j in range(split):
            fname = "{}_{}_{}.hdf5".format(ffeats, i, j)
            with tables.open_file(fname, 'r') as f:
                efeats.extend([fts for fts in f.root.features])
                ecentres.extend([cen for cen in f.root.centres])

    efeats = np.array(efeats)
    ecentres = np.array(ecentres)
    feats = np.array([dfeats[tuple(cen)] for cen in ecentres])

    assert len(dfeats) == len(efeats)
    assert np.all(feats == efeats)


def test_extractfeats_worker(make_shp_gtiff):

    _, ftif = make_shp_gtiff
    split = 2

    # return
    # TODO: The following just hangs!

    # Make pointspec
    ftif_json = os.path.splitext(ftif)[0] + ".json"
    pointspec.callback(ftif_json, geotiff=ftif, resolution=None, bbox=None,
                       pointlist=None, quiet=False)

    predis = None
    pworker = None

    # Start redis
    try:
        redisargs = ["redis-server", "--port", "6379"]
        predis = subprocess.Popen(redisargs, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)

        _proc_ready(predis)

        # Start the worker
        pworker = subprocess.Popen("uncoverml-worker", stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        _proc_ready(pworker)

        # Extract features from gtiff
        ffeats = os.path.splitext(ftif)[0] + "_worker"
        extractfeats.callback(geotiff=ftif, pointspec=ftif_json,
                              outfile=ffeats, patchsize=0, splits=split,
                              quiet=True, redisdb=0, redishost='localhost',
                              redisport=6379, standalone=False)
    finally:
        # Kill worker
        if predis is not None:
            predis.terminate()
        if pworker is not None:
            pworker.terminate()

    # Check all files created successfully
    for i in range(split):
        for j in range(split):
            fname = "{}_{}_{}.hdf5".format(ffeats, j, i)
            assert os.path.exists(fname)


def _proc_ready(proc, waitime=10):

    nbsr = io.NonBlockingStreamReader(proc.stdout)

    for i in range(waitime):
        try:
            for line in iter(nbsr.readline, None):
                if "ready" in line.decode():
                    return
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            proc.terminate()
            raise

    raise RuntimeError("Process {} never ready!".format(proc.args))
