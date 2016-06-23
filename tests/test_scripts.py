import os
import tables
import time

import numpy as np
import shapefile
import rasterio
from click import Context

# from uncoverml import geoio
from uncoverml.scripts.maketargets import main as maketargets
# from uncoverml.scripts.cvindexer import main as cvindexer
from uncoverml.scripts.extractfeats import main as extractfeats


def test_make_targets(make_shp):

    fshp, _ = make_shp
    field = "lon"
    folds = 6

    ctx = Context(maketargets)
    ctx.forward(maketargets, shapefile=fshp, fieldname=field,
                folds=folds)

    fhdf5 = os.path.splitext(fshp)[0] + "_" + field + ".hdf5"

    assert os.path.exists(fhdf5)

    with tables.open_file(fhdf5, mode='r') as f:
        lon = f.root.targets.read()
        Longitude = f.root.Longitude.read().flatten()
        Latitude = f.root.Latitude.read().flatten()
        finds = f.root.FoldIndices.read()
        Latitude_sorted = f.root.Latitude_sorted.read()

    assert np.allclose(lon, Longitude)

    # Test ordering
    latorder = [l1 <= l2 for l1, l2 in zip(Latitude_sorted[:-1],
                                           Latitude_sorted[1:])]
    assert all(latorder)

    # Validate target order is consistent with shapefile
    f = shapefile.Reader(fshp)
    shpcoords = np.array([p.points[0] for p in f.shapes()])

    hdfcoords = np.vstack((Longitude, Latitude)).T
    assert np.allclose(shpcoords, hdfcoords)

    # Test we have the right number of folds
    assert finds.min() == 0
    assert finds.max() == (folds - 1)


def test_extractfeats(make_gtiff, make_ipcluster):

    ftif = make_gtiff
    chunks = make_ipcluster
    outdir = os.path.dirname(ftif)
    name = "fchunk_worker"

    # Extract features from gtiff
    ctx = Context(extractfeats)
    ctx.forward(extractfeats, geotiff=ftif, name=name, outputdir=outdir)

    ffiles = []
    exists = []
    for i in range(1, chunks + 1):
        fname = os.path.join(outdir, "{}.part{}of{}.hdf5".format(
            name, i, chunks))
        exists.append(os.path.exists(fname))
        ffiles.append(fname)

    assert all(exists)

    # Now compare extracted features to geotiff
    with rasterio.open(ftif, 'r') as f:
        I = np.transpose(f.read(), [2, 1, 0])

    efeats = []
    for fname in ffiles:
        print(fname)
        with tables.open_file(fname, 'r') as f:
            strip = [fts for fts in f.root.features]
            efeats.append(np.reshape(strip, (I.shape[0], -1, I.shape[2])))

    efeats = np.concatenate(efeats[::-1], axis=1)

    assert I.shape == efeats.shape
    assert np.allclose(I, efeats)


def test_extractfeats_targets(make_shp, make_gtiff, make_ipcluster):
    ftif = make_gtiff
    fshp, hdf5_filenames = make_shp
    outdir = os.path.dirname(fshp)
    name = "fpatch"

    # Make target file
    field = "lat"
    fshp_targets = os.path.splitext(fshp)[0] + "_" + field + ".hdf5"
    ctx = Context(maketargets)
    ctx.forward(maketargets, shapefile=fshp, fieldname=field,
                outfile=fshp_targets)

    # Extract features from gtiff
    ctx = Context(extractfeats)
    ctx.forward(extractfeats, geotiff=ftif, name=name, outputdir=outdir,
                targets=fshp_targets)

    # Get the 4 parts
    feat_list = []
    for fname in hdf5_filenames:
        # fname = name + ".part{}of4.hdf5".format(i)
        with tables.open_file(os.path.join(outdir, fname), 'r') as f:
            feat_list.append(f.root.features[:])
    feats = np.concatenate(feat_list, axis=0)

    # Read lats and lons from targets
    with tables.open_file(fshp_targets, mode='r') as f:
        lonlat = np.vstack((f.root.Longitude_sorted.read(),
                            f.root.Latitude_sorted.read())).T

    assert np.allclose(feats, lonlat)
