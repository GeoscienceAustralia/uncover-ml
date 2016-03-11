import numpy as np

from uncoverml import geom


def test_latlon2pix(make_raster):

    res, x_range, y_range, lons, lats = make_raster
    gspec = geom.GridPointSpec(x_range, y_range, res)

    lonq = 55.
    latq = -36.

    x = np.where(lonq <= lons)[0][0]
    y = np.where(latq <= lats)[0][0]

    xy = gspec.lonlat2pix(np.array([[lonq, latq]]))

    assert(xy[0, 0] == x)
    assert(xy[0, 1] == y)


def test_pix2latlon(make_raster):

    res, x_range, y_range, lons, lats = make_raster
    gspec = geom.GridPointSpec(x_range, y_range, res)

    xq = [0, 10, 15]
    yq = [0, 22, 3]
    xy = np.array([xq, yq]).T

    latlon = gspec.pix2latlon(xy)

    latlon_true = np.array([lons[xq], lats[yq]]).T
    assert(np.allclose(latlon, latlon_true))


def test_shp(make_shp_gtiff):

    fn = make_shp_gtiff
    print(fn)

    assert 0
