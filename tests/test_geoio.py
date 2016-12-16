import pytest
from affine import Affine
import numpy as np
import rasterio

from uncoverml import geoio
from uncoverml.image import Image

crs = rasterio.crs.CRS({'init': 'epsg:4326'})

@pytest.fixture(params=[1., 0.5, 1.5])
def pix_size_single(request):
    return request.param


@pytest.fixture(params=[-0.5, 0.0, 0.5])
def origin_point(request):
    return request.param


@pytest.fixture(params=[False])
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
    src = geoio.ArrayImageSource(masked_array, origin, crs, pix_size)

    if num_chunks > 1:
        if chunk_position == 'start':
            chunk_index = 0
        elif chunk_position == 'end':
            chunk_index = num_chunks - 1
        else:
            chunk_index = round(num_chunks/2)
        ispec = Image(src, chunk_idx=chunk_index, nchunks=num_chunks)
    else:
        ispec = Image(src)
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
        lonlats, vals, _ = geoio.load_shapefile(filename, str(i))
        assert np.all(lonlats == true_lonlats)
        assert all(vals == i)


def test_array_image_src():
    res_x = 1000
    res_y = 500
    data = np.transpose(np.mgrid[0:res_x, 0:res_y], axes=(1, 2, 0))
    masked_array = np.ma.array(data=data, mask=False)
    pix_size = (1., 1.)
    origin = (0., 0.)
    src = geoio.ArrayImageSource(masked_array, origin, crs, pix_size)
    x_min = 0
    x_max = 10
    y_min = 1
    y_max = 20
    data_new = src.data(x_min, x_max, y_min, y_max)
    data_orig = masked_array[x_min:x_max, :][:, y_min:y_max]
    assert np.all(data_new.data == data_orig.data)
    assert np.all(data_new.mask == data_orig.mask)


@pytest.fixture
def array_image_src(request):

    res_x = 1000
    res_y = 500
    x_range = (50, 80)
    y_range = (-40, -30)

    pixsize_x = (x_range[1] - x_range[0]) / res_x
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
                                    crs,
                                    (pixsize_x, pixsize_y))
    return im_src



def test_Image_data(array_image_src):
    true_data = array_image_src._data
    data = Image(array_image_src).data()
    assert np.all(true_data.data == data.data)
    assert np.all(true_data.mask == data.mask)


def test_Image_split(array_image_src, num_chunks):
    Ichunks = []
    I = array_image_src._data
    for i in range(num_chunks):
        chunk = Image(array_image_src, chunk_idx=i,
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
        img = Image(array_image_src, chunk_idx=i, nchunks=nchunks,
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
