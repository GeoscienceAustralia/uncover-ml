import rasterio
import numpy as np
import shapefile as shp
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


def test_points_from_shp(make_shp_gtiff):

    fshp, _ = make_shp_gtiff

    coords = geom.points_from_shp(fshp)

    f = shp.Reader(fshp)
    fdict = {fname[0]: i for i, fname in enumerate(f.fields[1:])}
    lonlat = np.array(f.records())[:, [fdict['lon'], fdict['lat']]]

    assert np.allclose(coords, lonlat)


def test_values_from_shp(make_shp_gtiff):

    fshp, _ = make_shp_gtiff

    for i in range(10):
        vals = geom.values_from_shp(fshp, str(i))
        assert all(vals == i)
