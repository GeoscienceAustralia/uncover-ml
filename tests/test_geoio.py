import rasterio
import numpy as np
import shapefile as shp
from affine import Affine

from uncoverml import geoio


def test_file_indices_okay():
    # test the correct case
    t1 = ["/path/to/file.part1.hdf5", 
          "/path/to/file.part0.hdf5", 
          "/path/to/file.part2.hdf5"]
    assert geoio.file_indices_okay(t1)
    
    # multiple features
    t2 = t1 + ["/some/other/path.part0.hdf5", 
               "/some/other/path.part1.hdf5",
               "/some/other/path.part2.hdf5"]
    assert geoio.file_indices_okay(t2)

    # wierd paths
    t3 = t2 + ["/my/name.part0.hdf5",
               "/oh/dear/name.part1.hdf5",
               "name.part2.hdf5"]
    assert geoio.file_indices_okay(t3)

    # missing data
    t4 = t3[:-1]
    assert not geoio.file_indices_okay(t4)

    # craaazy data
    t5 = t3 + ["extra_file.hdf5"]
    assert not geoio.file_indices_okay(t5)


def test_files_by_chunk():
    t1 = ["/path/to/file.part1.hdf5", 
          "/path/to/file.part0.hdf5", 
          "/path/to/file.part2.hdf5",
          "/some/to/zile.part0.hdf5", 
          "/other/to/zile.part1.hdf5", 
          "/path/to/zile.part2.hdf5"]

    r = geoio.files_by_chunk(t1)

    answer = {0:["/path/to/file.part0.hdf5", "/some/to/zile.part0.hdf5"],
              1:["/path/to/file.part1.hdf5", "/other/to/zile.part1.hdf5"],
              2:["/path/to/file.part2.hdf5", "/path/to/zile.part2.hdf5"]}

    assert r == answer


def test_grid_affine(make_shp_gtiff):

    _, ftif = make_shp_gtiff

    with rasterio.open(ftif, 'r') as f:
        A = f.affine
        cA = f.affine * Affine.translation(0.5, 0.5)

    ispec = geoio.Image(ftif)
    assert np.allclose(A, ispec.A)
    assert np.allclose(cA, ispec.cA)
    assert np.allclose(~A, ~ispec.A)
    assert np.allclose(~cA, ~ispec.cA)


def test_bounding_box(make_raster):

    res, x_bound, y_bound, lons, lats, Ao, Ap = make_raster

    class f:
        affine = Ao
        width = res[0]
        height = res[1]

    bx, by = geoio.bounding_box(f)
    assert np.allclose(x_bound, bx)
    assert np.allclose(y_bound, by)


def test_latlon2pix(make_raster, make_shp_gtiff):

    res, x_bound, y_bound, lons, lats, Ao, Ap = make_raster
    _, ftif = make_shp_gtiff
    ispec = geoio.Image(ftif)

    x = [1, 47, 81]
    y = [0, 23, 43]

    xy = ispec.lonlat2pix(np.array([lons[x], lats[y]]).T)

    assert all(xy[:, 0] == x)
    assert all(xy[:, 1] == y)


def test_pix2latlon(make_raster, make_shp_gtiff):

    res, x_bound, y_bound, lons, lats, Ao, Ap = make_raster
    _, ftif = make_shp_gtiff
    ispec = geoio.Image(ftif)

    xq = [0, 10, 15]
    yq = [0, 22, 3]
    xy = np.array([xq, yq]).T

    latlon = ispec.pix2latlon(xy)

    latlon_true = np.array([lons[xq], lats[yq]]).T
    assert(np.allclose(latlon, latlon_true))


def test_pix2latlon2latlon2pix(make_raster, make_shp_gtiff):

    res, x_bound, y_bound, lons, lats, Ao, Ap = make_raster
    _, ftif = make_shp_gtiff
    ispec = geoio.Image(ftif)

    xq = [0, 10, 15]
    yq = [0, 22, 3]
    xy = np.array([xq, yq]).T

    xy2 = ispec.lonlat2pix(ispec.pix2latlon(xy))

    assert np.allclose(xy, xy2)


def test_points_from_shp(make_shp_gtiff):

    fshp, _ = make_shp_gtiff

    coords = geoio.points_from_shp(fshp)

    f = shp.Reader(fshp)
    fdict = {fname[0]: i for i, fname in enumerate(f.fields[1:])}
    lonlat = np.array(f.records())[:, [fdict['lon'], fdict['lat']]]

    assert np.allclose(coords, lonlat)


def test_values_from_shp(make_shp_gtiff):

    fshp, _ = make_shp_gtiff

    for i in range(10):
        vals = geoio.values_from_shp(fshp, str(i))
        assert all(vals == i)


def test_Image_split(make_shp_gtiff):

    _, ftif = make_shp_gtiff

    nchunks = 4
    overlap = 3

    with rasterio.open(ftif, 'r') as f:
        Iorig = np.transpose(f.read(), [2, 1, 0])

    I = geoio.Image(ftif).data()

    assert Iorig.shape == I.shape
    assert np.all(Iorig == I)

    Ichunks = []
    for i in range(nchunks):
        chunk = geoio.Image(ftif, chunk_idx=i, nchunks=nchunks).data()
        Ichunks.append(chunk)

    Irecon = np.hstack(Ichunks)

    assert I.shape == Irecon.shape
    assert np.all(I == Irecon)

    Ichunks = []
    for i in range(nchunks):
        chunk = geoio.Image(ftif, chunk_idx=i, nchunks=nchunks,
                            overlap=overlap).data()
        if i == 0:
            Ichunks.append(chunk[:, 0:-overlap])
        elif i == (nchunks - 1):
            Ichunks.append(chunk[:, overlap:])
        else:
            Ichunks.append(chunk[:, overlap:-overlap])

    Irecon = np.hstack(Ichunks)

    assert I.shape == Irecon.shape
    assert np.all(I == Irecon)
