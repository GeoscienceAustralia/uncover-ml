import rasterio
import numpy as np
from affine import Affine

from uncoverml import geom


def test_grid_affine(make_shp_gtiff):

    _, ftif = make_shp_gtiff

    with rasterio.open(ftif, 'r') as f:
        A = f.affine * Affine.translation(0.5, 0.5)
        shpe = f.shape
        rnge = geom.bounding_box(f)

    gspec = geom.GridPointSpec(*rnge, (shpe[1], shpe[0]))
    assert np.allclose(A, gspec.A)
    assert np.allclose(geom._invert_affine(A), gspec.iA)


def test_latlon_pixel_centres(make_raster):

    res, x_bound, y_bound, lons, lats, Ao, Ap = make_raster

    class f:
        affine = Ao
        width = res[0]
        height = res[1]

    lns, lts = geom.lonlat_pixel_centres(f)
    assert np.allclose(lats, lts)
    assert np.allclose(lons, lns)


def test_bounding_box(make_raster):

    res, x_bound, y_bound, lons, lats, Ao, Ap = make_raster

    class f:
        affine = Ao
        width = res[0]
        height = res[1]

    bx, by = geom.bounding_box(f)
    assert np.allclose(x_bound, bx)
    assert np.allclose(y_bound, by)


def test_latlon2pix(make_raster):

    res, x_bound, y_bound, lons, lats, Ao, Ap = make_raster
    gspec = geom.GridPointSpec(x_bound, y_bound, res)

    x = [1, 47, 81]
    y = [0, 23, 43]

    xy = gspec.lonlat2pix(np.array([lons[x], lats[y]]).T)

    assert all(xy[:, 0] == x)
    assert all(xy[:, 1] == y)


def test_pix2latlon(make_raster):

    # res, lons, lats, A = make_raster
    res, x_bound, y_bound, lons, lats, Ao, Ap = make_raster
    gspec = geom.GridPointSpec(x_bound, y_bound, res)

    xq = [0, 10, 15]
    yq = [0, 22, 3]
    xy = np.array([xq, yq]).T

    latlon = gspec.pix2latlon(xy)

    latlon_true = np.array([lons[xq], lats[yq]]).T
    assert(np.allclose(latlon, latlon_true))


def test_shp(make_shp_gtiff):

    fn = make_shp_gtiff
    print(fn)

    assert 0  # TODO
