import pytest
import numpy as np
import shapefile as shp
import rasterio
from affine import Affine

timg = np.reshape(np.arange(1, 17), (4, 4))


@pytest.fixture
def make_patch_31():
    pwidth = 1
    pstride = 1

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
                        [14, 15, 16]]])

    tx = np.array([1, 1, 2, 2])
    ty = np.array([1, 2, 1, 2])

    return timg, pwidth, pstride, tpatch, tx, ty


@pytest.fixture
def make_patch_12():
    pwidth = 0
    pstride = 2

    # Test output patches, patch centres
    tpatch = np.array([[[1]], 
                       [[3]],
                       [[9]], 
                       [[11]]])

    tx = np.array([0,0,2,2])
    ty = np.array([0,2,0,2])

    return timg, pwidth, pstride, tpatch, tx, ty


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
                        [14, 15, 16]]])

    return timg, pwidth, points, tpatch


@pytest.fixture(params=[make_patch_31, make_patch_12])
def make_multi_patch(request):
    return request.param()


@pytest.fixture
def make_raster():

    res_x = 100
    res_y = 50
    x_range = (50, 80)
    y_range = (-40, -30)

    pix_x = (x_range[1] - x_range[0]) / res_x
    pix_y = (y_range[1] - y_range[0]) / res_y

    Aorig = Affine(pix_x, 0, x_range[0],
                   0, -pix_y, y_range[1])
    Apix = Aorig * Affine.translation(0.5, 0.5)

    lons = x_range[0] + (np.arange(res_x) + 0.5) * pix_x
    lats = y_range[1] - (np.arange(res_y) + 0.5) * pix_y

    # lons = np.arange(x_range[0] + pix_x / 2, x_range[1] + pix_x / 2, pix_x)
    # lats = np.arange(y_range[1] - pix_y / 2, y_range[0] - pix_y / 2, pix_y)

    x_bound = (x_range[0], x_range[1] + pix_x)
    y_bound = (y_range[0] - pix_y, y_range[1])

    return (res_x, res_y), x_bound, y_bound, lons, lats, Aorig, Apix


@pytest.fixture(scope='session')
def make_shp_gtiff(tmpdir_factory):

    # File names for test shapefile and test geotiff
    fshp = str(tmpdir_factory.mktemp('shapes').join('test').realpath())
    ftif = str(tmpdir_factory.mktemp('tif').join('test.tif').realpath())

    # Create grid
    res, x_bound, y_bound, lons, lats, Ao, Ap = make_raster()

    # Generate data for shapefile
    nsamples = 100
    ntargets = 10
    dlon = x_bound[0] + np.random.rand(nsamples) * (x_bound[1] - x_bound[0])
    dlat = y_bound[0] + np.random.rand(nsamples) * (y_bound[1] - y_bound[0])
    fields = [str(i) for i in range(ntargets)] + ["lon", "lat"]
    vals = np.ones((nsamples, ntargets)) * np.arange(ntargets)
    vals = np.hstack((vals, np.array([dlon, dlat]).T))

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

    w.save(fshp)

    # Generate data for geotiff
    lons -= (lons[1] - lons[0]) / 2.  # Undo pixel centring
    lats += (lats[1] - lats[0]) / 2.
    Lons, Lats = np.meshgrid(lons, lats)

    # Write geotiff
    profile = {'driver': "GTiff",
               'width': len(lons),
               'height': len(lats),
               'count': 2,
               'dtype': rasterio.float64,
               'transform': Ao,
               'crs': {'proj': 'longlat',
                       'ellps': 'WGS84',
                       'datum': 'WGS84',
                       'nodefs': True
                       }
               }

    with rasterio.open(ftif, 'w', **profile) as f:
        f.write(np.array([Lons, Lats]))

    return fshp, ftif
