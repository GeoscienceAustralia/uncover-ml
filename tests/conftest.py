from __future__ import division

import random
import string
import os.path

import pytest
import numpy as np
import shapefile as shp
from uncoverml import mpiops

timg = np.reshape(np.arange(1, 17), (4, 4, 1))


@pytest.fixture
def random_filename(tmpdir_factory):
    dir = str(tmpdir_factory.mktemp('uncoverml').realpath())
    fname = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
    filename = os.path.join(dir, fname)
    return filename


@pytest.fixture
def masked_array():
    x = np.ma.masked_array(data=[[0., 1.], [2., 3.], [4., 5.]],
                           mask=[[True, False], [False, True], [False, False]],
                           fill_value=1e23)
    return x


@pytest.fixture
def int_masked_array():
    x = np.ma.masked_array(data=[[1, 1], [2, 3], [4, 5], [3, 3]],
                           mask=[[True, False], [False, True], [False, False],
                                 [False, False]],
                           fill_value=-9999, dtype=int)
    return x


@pytest.fixture
def make_patch_31():
    pwidth = 1

    # Test output patches, patch centres
    tpatch = np.array([[[1, 2, 3],
                        [5, 6, 7],
                        [9, 10, 11]],
                       [[2, 3, 4],
                        [6, 7, 8],
                        [10, 11, 12]],
                       [[5, 6, 7],
                        [9, 10, 11],
                        [13, 14, 15]],
                       [[6, 7, 8],
                        [10, 11, 12],
                        [14, 15, 16]]])[:, :, :, np.newaxis]

    tx = np.array([1, 1, 2, 2])
    ty = np.array([1, 2, 1, 2])

    return timg, pwidth, tpatch, tx, ty


@pytest.fixture
def make_patch_11():
    pwidth = 0

    # Test output patches, patch centres
    tpatch = np.array([[[timg.flatten()]]]).T

    tx, ty = [g.flatten() for g in np.meshgrid(np.arange(3), np.arange(3))]

    return timg, pwidth, tpatch, tx, ty


@pytest.fixture
def make_points():
    pwidth = 1
    points = np.array([[1, 1], [2, 1], [2, 2]])

    tpatch = np.array([[[1, 2, 3],
                        [5, 6, 7],
                        [9, 10, 11]],
                       [[5, 6, 7],
                        [9, 10, 11],
                        [13, 14, 15]],
                       [[6, 7, 8],
                        [10, 11, 12],
                        [14, 15, 16]]])[:, :, :, np.newaxis]

    return timg, pwidth, points, tpatch


@pytest.fixture(params=[make_patch_31, make_patch_11])
def make_multi_patch(request):
    return request.param()

@pytest.fixture
def shapefile(random_filename, request):

    # File names for test shapefile and test geotiff
    filename = random_filename + ".shp"
    lons = np.arange(0, 20, 2)
    lats = np.arange(-10, 30, 2)

    # Generate data for shapefile
    nsamples = 100
    ntargets = 10
    dlon = lons[np.random.randint(0, high=len(lons), size=nsamples)]
    dlat = lats[np.random.randint(0, high=len(lats), size=nsamples)]
    fields = [str(i) for i in range(ntargets)] + ["lon", "lat"]
    vals = np.ones((nsamples, ntargets)) * np.arange(ntargets)
    lonlats = np.array([dlon, dlat]).T
    vals = np.hstack((vals, lonlats))

    # write shapefile
    w = shp.Writer(shp.POINT)
    w.autoBalance = 1

    # points
    for p in zip(dlon, dlat):
        w.point(*p)

    # fields
    for f in fields:
        w.field(f, 'N', 16, 6)

    # records
    for v in vals:
        vdict = dict(zip(fields, v))
        w.record(**vdict)

    w.save(filename)

    return lonlats, filename


@pytest.fixture
def linear_data():

    Nt = 800
    Ns = 200
    x = np.linspace(-2, 2, Nt + Ns)
    y = 3 + 2 * x
    X = x[:, np.newaxis]

    trind = np.random.choice(Nt + Ns, Nt, replace=False)
    tsind = np.zeros_like(x, dtype=bool)
    tsind[trind] = True
    tsind = np.where(~tsind)[0]

    return y[trind], X[trind], y[tsind], X[tsind]


# Make sure all MPI tests use this fixure
@pytest.fixture()
def mpisync(request):
    mpiops.comm.barrier()

    def fin():
        mpiops.comm.barrier()
    request.addfinalizer(fin)
    return mpiops.comm