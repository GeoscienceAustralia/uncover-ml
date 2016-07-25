import os
import pytest
import tables as hdf
from affine import Affine
import numpy as np

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


@pytest.fixture(params=[1., 0.5, 1.5])
def pix_size_single(request):
    return request.param


@pytest.fixture(params=[-0.5, 0.0, 0.5])
def origin_point(request):
    return request.param


@pytest.fixture(params=[True, False])
def is_flipped(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def overlap_pixels(request):
    return request.param


@pytest.fixture(params=[1, 2, 9])
def num_chunks(request):
    return request.param


@pytest.fixture(params=['start', 'middle', 'end'])
def chunk_position(request):
    return request.param


def make_image(pix_size_single, origin_point,
               is_flipped, num_chunks, chunk_position):

    data = np.random.rand(32, 24, 1)
    masked_array = np.ma.array(data=data, mask=False)
    pix_size = (pix_size_single, pix_size_single)
    if is_flipped:
        pix_size = (pix_size_single, -1 * pix_size_single)
    # origin needs to change as well
    origin = (origin_point, origin_point)
    src = geoio.ArrayImageSource(masked_array, origin, pix_size)

    if num_chunks > 1:
        if chunk_position == 'start':
            chunk_index = 0
        elif chunk_position == 'end':
            chunk_index = num_chunks - 1
        else:
            chunk_index = round(num_chunks/2)
        ispec = geoio.Image(src, chunk_idx=chunk_index, nchunks=num_chunks)
    else:
        ispec = geoio.Image(src)
    return ispec


def test_latlon2pix_edges(pix_size_single, origin_point, is_flipped,
                          num_chunks, chunk_position):

    img = make_image(pix_size_single, origin_point, is_flipped,
                     num_chunks, chunk_position)
    chunk_idx = img.chunk_idx
    res_x = img._full_res[0]
    res_y = img._full_res[1]
    pix_size = (img.pixsize_x, img.pixsize_y)
    origin = (img._start_lon, img._start_lat)

    # compute chunks
    lons = np.arange(res_x + 1) * pix_size[0] + origin[0]  # right edge +1
    all_lats = np.arange(res_y) * pix_size[1] + origin[1]
    lats_chunks = np.array_split(all_lats, num_chunks)[chunk_idx]
    pix_x = np.concatenate((np.arange(res_x), [res_x - 1]))
    pix_y_chunks = range(lats_chunks.shape[0])
    if chunk_position == 'end':
        pix_y = np.concatenate((pix_y_chunks, [pix_y_chunks[-1]]))
        lats = np.concatenate((lats_chunks, [res_y * pix_size[1] + origin[1]]))
    else:
        pix_y = pix_y_chunks
        lats = lats_chunks

    d = np.array([[a, b] for a in lons for b in lats])
    xy = img.lonlat2pix(d)
    true_xy = np.array([[a, b] for a in pix_x for b in pix_y])
    assert np.all(xy == true_xy)


def test_latlon2pix_internals(pix_size_single, origin_point, is_flipped,
                              num_chunks, chunk_position):

    img = make_image(pix_size_single, origin_point, is_flipped,
                     num_chunks, chunk_position)
    chunk_idx = img.chunk_idx
    res_x = img._full_res[0]
    res_y = img._full_res[1]
    pix_size = (img.pixsize_x, img.pixsize_y)
    origin = (img._start_lon, img._start_lat)

    # +0.5 for centre of pixels
    lons = (np.arange(res_x) + 0.5) * pix_size[0] + origin[0]
    all_lats = (np.arange(res_y) + 0.5) * pix_size[1] + origin[1]
    lats = np.array_split(all_lats, num_chunks)[chunk_idx]

    pix_x = np.arange(res_x)
    pix_y = np.arange(lats.shape[0])

    d = np.array([[a, b] for a in lons for b in lats])
    xy = img.lonlat2pix(d)
    true_xy = np.array([[a, b] for a in pix_x for b in pix_y])
    assert np.all(xy == true_xy)


def test_pix2latlong(pix_size_single, origin_point, is_flipped,
                     num_chunks, chunk_position):

    img = make_image(pix_size_single, origin_point, is_flipped,
                     num_chunks, chunk_position)
    chunk_idx = img.chunk_idx
    res_x = img._full_res[0]
    res_y = img._full_res[1]
    pix_size = (img.pixsize_x, img.pixsize_y)
    origin = (img._start_lon, img._start_lat)

    true_lons = np.arange(res_x) * pix_size[0] + origin[0]
    all_lats = np.arange(res_y) * pix_size[1] + origin[1]
    true_lats = np.array_split(all_lats, num_chunks)[chunk_idx]
    true_d = np.array([[a, b] for a in true_lons for b in true_lats])

    pix_x = np.arange(res_x)
    pix_y = np.arange(img.resolution[1])  # chunk resolution

    xy = np.array([[a, b] for a in pix_x for b in pix_y])

    lonlats = img.pix2lonlat(xy)
    assert np.all(lonlats == true_d)


def test_load_shapefile(shapefile):
    true_lonlats, filename = shapefile
    for i in range(10):
        lonlats, vals = geoio.load_shapefile(filename, str(i))
        assert np.all(lonlats == true_lonlats)
        assert all(vals == i)


def test_array_image_src():
    res_x = 1000
    res_y = 500
    data = np.transpose(np.mgrid[0:res_x, 0:res_y], axes=(1, 2, 0))
    masked_array = np.ma.array(data=data, mask=False)
    pix_size = (1., 1.)
    origin = (0., 0.)
    src = geoio.ArrayImageSource(masked_array, origin, pix_size)
    x_min = 0
    x_max = 10
    y_min = 1
    y_max = 20
    data_new = src.data(x_min, x_max, y_min, y_max)
    data_orig = masked_array[x_min:x_max, :][:, y_min:y_max]
    assert np.all(data_new.data == data_orig.data)
    assert np.all(data_new.mask == data_orig.mask)


@pytest.fixture(params=[True, False])
def array_image_src(request):

    res_x = 1000
    res_y = 500
    x_range = (50, 80)
    y_range = (-40, -30)

    flipped = request.param
    if flipped:
        y_range = (-30, -40)

    pixsize_x = (x_range[1] - x_range[0]) / res_x
    # Try both flipped and non_flipped images
    pixsize_y = (y_range[1] - y_range[0]) / res_y

    A = Affine(pixsize_x, 0, x_range[0],
               0, -pixsize_y, y_range[1])

    lons = np.array([(x, 0) * A for x in np.arange(res_x)])[:, 0]
    lats = np.array([(0, y) * A for y in np.arange(res_y)])[:, 1]

    channel_1 = lons[:, np.newaxis] * np.ones((res_x, res_y))
    channel_2 = lats[np.newaxis, :] * np.ones((res_x, res_y))
    data = np.concatenate((channel_1[:, :, np.newaxis],
                           channel_2[:, :, np.newaxis]), axis=2)
    masked_array = np.ma.MaskedArray(data=data, mask=False)
    im_src = geoio.ArrayImageSource(masked_array,
                                    (x_range[0], y_range[0]),
                                    (pixsize_x, pixsize_y))
    return im_src



def test_Image_data(array_image_src):
    true_data = array_image_src._data
    data = geoio.Image(array_image_src).data()
    assert np.all(true_data.data == data.data)
    assert np.all(true_data.mask == data.mask)


def test_Image_split(array_image_src, num_chunks):
    Ichunks = []
    I = array_image_src._data
    for i in range(num_chunks):
        chunk = geoio.Image(array_image_src, chunk_idx=i,
                            nchunks=num_chunks).data()
        Ichunks.append(chunk)

    # Reverse Ichunks to account for y-decreasing convention in images
    Irecon = np.hstack(Ichunks)

    assert I.shape == Irecon.shape
    assert np.all(I == Irecon)


def test_Image_split_overlap(array_image_src, num_chunks, overlap_pixels):
    nchunks = num_chunks
    overlap = overlap_pixels
    I = array_image_src._data
    Ichunks = []
    images = []
    for i in range(nchunks):
        img = geoio.Image(array_image_src, chunk_idx=i, nchunks=nchunks,
                          overlap=overlap)
        images.append(img)
        chunk = img.data()
        if num_chunks == 1:
            Ichunks.append(chunk)
        elif i == 0:
            Ichunks.append(chunk[:, :-overlap])
        elif i == (nchunks - 1):
            Ichunks.append(chunk[:, overlap:])
        else:
            Ichunks.append(chunk[:, overlap:-overlap])

    Irecon = np.hstack(Ichunks)
    assert I.shape == Irecon.shape
    assert np.all(I == Irecon)


def test_points_fromto_hdf(random_filename):

    filename = random_filename + ".hdf5"

    data = {"intfield": np.arange(10, dtype=int),
            "floatfield": np.random.rand(2, 3),
            "boolfield": np.random.choice(2, 5).astype(bool)}
    geoio.points_to_hdf(filename, data)
    new_data = geoio.points_from_hdf(filename, list(data.keys()))
    matches = [k for k in data if np.all(data[k] == new_data[k])]
    assert set(matches) == set(data.keys())
    os.remove(filename)


def test_load_attributes(random_filename):
    filename = random_filename + ".hdf5"
    shape = (10, 4)
    bbox = np.array([1.0, 2.0, 1.0, 2.0])
    with hdf.open_file(filename, 'w') as f:
        f.create_carray("/", "features",
                             atom=hdf.Float64Atom(),
                             obj=np.random.rand(10, 4))
        f.root._v_attrs.image_shape = shape
        f.root._v_attrs.image_bbox = bbox

    fdict = {0: [filename]}
    nshape, nbbox = geoio.load_attributes(fdict)
    assert shape == nshape
    assert np.all(bbox == nbbox)
    os.remove(filename)


def test_load_attributes_blank(random_filename):
    filename = random_filename + ".hdf5"
    with hdf.open_file(filename, 'w') as f:
            f.create_carray("/", "features",
                                 atom=hdf.Float64Atom(),
                                 obj=np.random.rand(10, 4))

    fdict = {0: [filename]}
    shape, bbox = geoio.load_attributes(fdict)
    assert shape is None
    assert bbox is None
    os.remove(filename)


def test_output_filename():
    true_filename = "/path/to/my/file/featurename.part3of5.hdf5"
    filename = geoio.output_filename("featurename", 2, 5, "/path/to/my/file")
    assert true_filename == filename


def test_output_blank(random_filename):
    filename = random_filename + ".hdf5"
    geoio.output_blank(filename)
    with hdf.open_file(filename, mode='r') as f:
        assert f.root._v_attrs["blank"]
    os.remove(filename)


def test_output_features(random_filename):
    filename = random_filename + ".hdf5"
    shape = (100, 5)
    feature_vector_data = np.random.random(size=shape)
    feature_vector_mask = np.random.choice(2, size=shape).astype(bool)
    X = np.ma.array(data=feature_vector_data, mask=feature_vector_mask)
    shape = (20, 5, 3)
    bbox = np.array([0.0, 1.0, 1.0, 0.0])
    geoio.output_features(X, filename, featname="features",
                          shape=shape, bbox=bbox)

    with hdf.open_file(filename, mode='r') as f:
        assert not f.root._v_attrs["blank"]
        assert np.all(f.root.mask.read() == X.mask)
        assert np.all(f.root.features.read() == X.data)
        assert np.all(f.root._v_attrs.image_bbox == bbox)
        assert np.all(f.root._v_attrs.image_shape == shape)
    os.remove(filename)


def test_load_and_cat(random_filename):
    filename1 = random_filename + "_1.hdf5"
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
    filename2 = random_filename + "_2.hdf5"
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
    os.remove(filename1)
    os.remove(filename2)
