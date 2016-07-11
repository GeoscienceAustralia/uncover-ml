import rasterio
import numpy as np
import shapefile as shp
import os.path
import tables as hdf

from uncoverml import geoio


def test_file_indices_okay():
    # test the correct case
    t1 = ["/path/to/file.part1of3.hdf5",
          "/path/to/file.part3of3.hdf5",
          "/path/to/file.part2of3.hdf5"]
    assert geoio.file_indices_okay(t1)

    # multiple features
    t2 = t1 + ["/some/other/path.part3of3.hdf5",
               "/some/other/path.part1of3.hdf5",
               "/some/other/path.part2of3.hdf5"]
    assert geoio.file_indices_okay(t2)

    # wierd paths
    t3 = t2 + ["/my/name.part3of3.hdf5",
               "/oh/dear/name.part1of3.hdf5",
               "name.part2of3.hdf5"]
    assert geoio.file_indices_okay(t3)

    # missing data
    t4 = t3[:-1]
    assert not geoio.file_indices_okay(t4)

    # craaazy data
    t5 = t3 + ["extra_fileof3.hdf5"]
    assert not geoio.file_indices_okay(t5)


def test_files_by_chunk():
    t1 = ["/path/to/file.part1of3.hdf5",
          "/path/to/file.part3of3.hdf5",
          "/path/to/file.part2of3.hdf5",
          "/some/to/zile.part1of3.hdf5",
          "/other/to/zile.part2of3.hdf5",
          "/path/to/zile.part3of3.hdf5"]

    r = geoio.files_by_chunk(t1)

    answer = {0: ["/path/to/file.part1of3.hdf5",
                  "/some/to/zile.part1of3.hdf5"],
              1: ["/path/to/file.part2of3.hdf5",
                  "/other/to/zile.part2of3.hdf5"],
              2: ["/path/to/file.part3of3.hdf5",
                  "/path/to/zile.part3of3.hdf5"]}

    assert r == answer


def test_latlon2pix_edges(make_raster, make_gtiff):

    res, x_bound, y_bound, lons, lats, Ao = make_raster
    ftif = make_gtiff
    ispec = geoio.Image(ftif)

    x = [1, 47, 81]
    y = [0, 23, 43]

    xy = ispec.lonlat2pix(np.array([lons[x], lats[y]]).T)

    assert all(xy[:, 0] == x)
    assert all(xy[:, 1] == y)


def test_latlon2pix_internals(make_raster, make_gtiff):

    res, x_bound, y_bound, lons, lats, Ao = make_raster
    ftif = make_gtiff
    ispec = geoio.Image(ftif)

    x = [1, 47, 81]
    y = [0, 23, 43]

    xy = ispec.lonlat2pix(np.array([lons[x] + 0.5 * ispec.pixsize_x,
                          lats[y] + 0.5 * ispec.pixsize_y]).T)

    assert all(xy[:, 0] == x)
    assert all(xy[:, 1] == y)


def test_pix2lonlat(make_raster, make_gtiff):

    res, x_bound, y_bound, lons, lats, Ao = make_raster
    ftif = make_gtiff
    ispec = geoio.Image(ftif)

    xq = [0, 10, 15]
    yq = [0, 22, 3]
    xy = np.array([xq, yq]).T

    latlon = ispec.pix2lonlat(xy)

    latlon_true = np.array([lons[xq], lats[yq]]).T
    assert(np.allclose(latlon, latlon_true))


def test_pix2lonlat2latlon2pix(make_raster, make_gtiff):

    res, x_bound, y_bound, lons, lats, Ao = make_raster
    ftif = make_gtiff
    ispec = geoio.Image(ftif)

    xq = [0, 10, 15]
    yq = [0, 22, 3]
    xy = np.array([xq, yq]).T

    xy2 = ispec.lonlat2pix(ispec.pix2lonlat(xy))

    assert np.allclose(xy, xy2)


def test_points_from_shp(make_shp):

    fshp, _ = make_shp

    coords = geoio.points_from_shp(fshp)

    f = shp.Reader(fshp)
    fdict = {fname[0]: i for i, fname in enumerate(f.fields[1:])}
    lonlat = np.array(f.records())[:, [fdict['lon'], fdict['lat']]]

    assert np.allclose(coords, lonlat)


def test_values_from_shp(make_shp):

    fshp, _ = make_shp

    for i in range(10):
        vals = geoio.values_from_shp(fshp, str(i))
        assert all(vals == i)


def test_Image_split(make_gtiff):

    ftif = make_gtiff

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

    # Reverse Ichunks to account for y-decreasing convention in images
    Irecon = np.hstack(Ichunks[::-1])

    assert I.shape == Irecon.shape
    assert np.all(I == Irecon)

    Ichunks = []
    for i in range(nchunks):
        # Reverse Ichunks to account for y-decreasing convention in images
        i = nchunks - i - 1
        chunk = geoio.Image(ftif, chunk_idx=i, nchunks=nchunks,
                            overlap=overlap).data()
        if i == 0:
            Ichunks.append(chunk[:, overlap:])
        elif i == (nchunks - 1):
            Ichunks.append(chunk[:, 0:-overlap])
        else:
            Ichunks.append(chunk[:, overlap:-overlap])

    Irecon = np.hstack(Ichunks)

    assert I.shape == Irecon.shape
    assert np.all(I == Irecon)


def test_points_fromto_hdf(make_tmpdir):

    filename = os.path.join(make_tmpdir, "fromto.hdf5")

    data = {"intfield": np.arange(10, dtype=int),
            "floatfield": np.random.rand(2, 3),
            "boolfield": np.random.choice(2, 5).astype(bool)}
    geoio.points_to_hdf(filename, data)
    new_data = geoio.points_from_hdf(filename, list(data.keys()))
    matches = [k for k in data if np.all(data[k] == new_data[k])]
    assert set(matches) == set(data.keys())


def test_load_attributes(make_tmpdir):
    filename = os.path.join(make_tmpdir, "loadattribs.hdf5")
    shape = (10, 4)
    bbox = np.array([1.0, 2.0, 1.0, 2.0])
    with hdf.open_file(filename, 'w') as f:
        f.create_carray("/", "features",
                             atom=hdf.Float64Atom(),
                             obj=np.random.rand(10, 4))
        f.root.features.attrs.shape = shape
        f.root.features.attrs.bbox = bbox

    fdict = {0: [filename]}
    nshape, nbbox = geoio.load_attributes(fdict)
    assert shape == nshape
    assert np.all(bbox == nbbox)


def test_load_attributes_blank(make_tmpdir):
    filename = os.path.join(make_tmpdir, "loadattribs_blank.hdf5")
    with hdf.open_file(filename, 'w') as f:
            f.create_carray("/", "features",
                                 atom=hdf.Float64Atom(),
                                 obj=np.random.rand(10, 4))

    fdict = {0: [filename]}
    shape, bbox = geoio.load_attributes(fdict)
    assert shape is None
    assert bbox is None


def output_filename(feature_name, chunk_index, n_chunks, output_dir):
    filename = feature_name + ".part{}of{}.hdf5".format(chunk_index + 1,
                                                        n_chunks)
    full_path = os.path.join(output_dir, filename)
    return full_path


def test_output_filename():
    true_filename = "/path/to/my/file/featurename.part3of5.hdf5"
    filename = geoio.output_filename("featurename", 2, 5, "/path/to/my/file")
    assert true_filename == filename


def test_output_blank(make_tmpdir):
    filename = os.path.join(make_tmpdir, "outputblank.hdf5")
    geoio.output_blank(filename)
    with hdf.open_file(filename, mode='r') as f:
        assert f.root._v_attrs["blank"]


def test_output_features(make_tmpdir):
    filename = os.path.join(make_tmpdir, "outputfeatures.hdf5")
    shp = (100, 5)
    feature_vector_data = np.random.random(size=shp)
    feature_vector_mask = np.random.choice(2, size=shp).astype(bool)
    X = np.ma.array(data=feature_vector_data, mask=feature_vector_mask)
    shape = (20, 5, 3)
    bbox = np.array([0.0, 1.0, 1.0, 0.0])
    geoio.output_features(X, filename, featname="features",
                          shape=shape, bbox=bbox)

    with hdf.open_file(filename, mode='r') as f:
        assert not f.root._v_attrs["blank"]
        assert np.all(f.root.mask.read() == X.mask)
        assert np.all(f.root.features.read() == X.data)
        assert np.all(f.root.mask.attrs.bbox == bbox)
        assert np.all(f.root.features.attrs.bbox == bbox)
        assert np.all(f.root.mask.attrs.shape == shape)
        assert np.all(f.root.features.attrs.shape == shape)


def test_load_and_cat(make_tmpdir):
    filename1 = os.path.join(make_tmpdir, "loadcat1.hdf5")
    data1 = np.random.rand(10, 4)
    mask1 = (np.random.rand(10, 4) > 0.5).astype(bool)
    with hdf.open_file(filename1, 'w') as f:
            f.root._v_attrs["blank"] = False
            f.create_carray("/", "features",
                                 atom=hdf.Float64Atom(),
                                 obj=data1)
            f.create_carray("/", "mask",
                                 atom=hdf.BoolAtom(),
                                 obj=mask1)
    filename2 = os.path.join(make_tmpdir, "loadcat2.hdf5")
    data2 = np.random.rand(10, 2)
    mask2 = (np.random.rand(10, 2) > 0.5).astype(bool)
    with hdf.open_file(filename2, 'w') as f:
            f.root._v_attrs["blank"] = False
            f.create_carray("/", "features",
                                 atom=hdf.Float64Atom(),
                                 obj=data2)
            f.create_carray("/", "mask",
                                 atom=hdf.BoolAtom(),
                                 obj=mask2)
    x = geoio.load_and_cat([filename1, filename2])
    trueval = np.concatenate((data1, data2), axis=1)
    truemask = np.concatenate((mask1, mask2), axis=1)
    assert np.all(x.data == trueval)
    assert np.all(x.mask == truemask)
