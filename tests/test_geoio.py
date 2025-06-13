import pytest
import numpy as np
import rasterio
import os
import tempfile
from unittest.mock import MagicMock, patch
from affine import Affine
from types import SimpleNamespace
from collections import OrderedDict
from rasterio.transform import from_origin
from uncoverml import geoio
from uncoverml.image import Image
from uncoverml.geoio import RasterioImageSource, ImageWriter


crs = "EPSG:4326"

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


@pytest.fixture
def make_geotiff(tmp_path):
    width, height = 10, 20
    count = 1
    dtype = 'float32'
    nodata = -9999.0
    transform = from_origin(0, 100, 1, 1)

    filepath = tmp_path / 'test.tif'
    data = np.arange(width * height, dtype=dtype).reshape((height, width))

    with rasterio.open(
        filepath, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data, 1)

    return str(filepath), data, nodata, (width, height)


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


@patch('uncoverml.geoio.plt')
@patch('uncoverml.geoio.log')
def test_add_groups_with_grouping(mock_log, mock_plt):
    lonlat = np.array([[10.0, -20.0], [10.1, -20.1], [10.2, -20.2]])
    grouping_data = np.array(['A', 'A', 'B'])
    conf = MagicMock()
    conf.group_col = 'region'
    conf.target_groups_file = '/tmp/test_groups.png'
    mock_plt.rcParams = {
        'axes.prop_cycle': MagicMock(by_key=MagicMock(return_value={'color': ['#ff0000', '#00ff00', '#0000ff']}))
    }

    groups = geoio.add_groups(lonlat, grouping_data, conf)
    assert isinstance(groups, np.ndarray)
    assert set(groups) == {0, 1}
    mock_plt.savefig.assert_called_once()


@patch('uncoverml.geoio.plt')
@patch('uncoverml.geoio.DBSCAN')
@patch('uncoverml.geoio.log')
def test_add_groups_without_grouping(mock_log, mock_dbscan, mock_plt):
    lonlat = np.random.rand(10, 2)
    conf = MagicMock()
    conf.group_col = None
    conf.groups_eps = 0.5
    conf.target_groups_file = '/tmp/test_groups_dbscan.png'

    mock_dbscan_instance = MagicMock()
    mock_dbscan_instance.labels_ = np.array([0] * 5 + [1] * 5)
    mock_dbscan.return_value = mock_dbscan_instance
    mock_plt.rcParams = {
        'axes.prop_cycle': MagicMock(
            by_key=MagicMock(return_value={'color': ['#ff0000', '#00ff00', '#0000ff']})
        )
    }

    groups = geoio.add_groups(lonlat, None, conf)
    assert set(groups) == {0, 1}
    mock_plt.scatter.assert_called_once()


@patch('uncoverml.geoio.load_shapefile')
@patch('uncoverml.geoio.add_groups')
@patch('uncoverml.geoio.mpiops')
@patch('uncoverml.geoio.log')
def test_load_targets(mock_log, mock_mpiops, mock_add_groups, mock_load_shapefile):
    conf = MagicMock()
    conf.group_targets = True
    conf.group_col = 'group_id'
    conf.weighted_model = False
    lonlat = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = np.array([5, 6])
    other = {'group_id': np.array(['A', 'B'])}
    mock_load_shapefile.return_value = (lonlat, values, other)
    mock_add_groups.return_value = np.array([0, 1])
    mock_comm = MagicMock()
    mock_comm.scatter.side_effect = lambda x, root=0: np.array(x[0]) if isinstance(x, list) else x
    mock_mpiops.comm = mock_comm
    mock_mpiops.chunk_index = 0
    mock_mpiops.chunks = 1

    targets = geoio.load_targets('fake.shp', 'val', conf)
    assert targets.groups.shape == (2,)
    assert targets.weights.shape == (2,)


@patch('uncoverml.geoio.image.Image')
@patch('uncoverml.geoio.RasterioImageSource.__init__', return_value=None)
def test_get_image_spec_from_nchannels(mock_rio_init, mock_image):
    config = MagicMock()
    config.prediction_template = None
    config.is_prediction = False
    config.feature_sets = [MagicMock(files=['fake.tif'])]
    mock_image_instance = MagicMock()
    mock_image_instance.patched_shape.return_value = (100, 100)
    mock_image_instance.patched_bbox.return_value = ((0, 0), (1, 1))
    mock_image_instance.crs = 'EPSG:4326'
    mock_image.return_value = mock_image_instance

    shape, bbox, crs = geoio.get_image_spec_from_nchannels(2, config)

    assert shape == (100, 100, 2)
    assert bbox == ((0, 0), (1, 1))
    assert crs == 'EPSG:4326'


@patch('uncoverml.geoio.get_image_spec_from_nchannels')
def test_get_image_spec(mock_get_spec):
    config = MagicMock()
    model = MagicMock()
    model.get_predict_tags.return_value = ['a', 'b', 'c']
    mock_get_spec.return_value = ((10, 10, 3), ((0, 0), (1, 1)), 'EPSG:4326')

    result = geoio.get_image_spec(model, config)

    assert result[0] == (10, 10, 3)
    mock_get_spec.assert_called_once_with(3, config)


def test_rasterio_image_source_init(make_geotiff):
    filepath, _, nodata, (width, height) = make_geotiff
    src = RasterioImageSource(filepath)

    assert src.filename == filepath
    assert src.nodata_value == nodata
    assert src.full_resolution == (width, height, 1)
    assert src.pixsize_x > 0
    assert src.pixsize_y > 0
    assert src.origin_longitude == 0
    assert src.origin_latitude == 80


def test_rasterio_image_source_data_shape(make_geotiff):
    filepath, data, _, _ = make_geotiff
    src = RasterioImageSource(filepath)

    out = src.data(0, 5, 0, 10)
    assert out.shape == (5, 10, 1)
    assert isinstance(out, np.ma.MaskedArray)


@patch('uncoverml.geoio.mpiops')
@patch('uncoverml.geoio.rasterio.open')
def test_imagewriter(mock_rasterio_open, mock_mpiops):
    mock_mpiops.chunk_index = 0
    mock_mpiops.chunks = 1
    mock_mpiops.comm.bcast = lambda x, root=0: x
    mock_raster = MagicMock()
    mock_rasterio_open.return_value = mock_raster
    shape = (10, 10)
    bbox = np.array([[0, 0], [1, 1]])
    crs = 'EPSG:4326'
    band_tags = ['Band1']
    outputdir = tempfile.mkdtemp()
    writer = ImageWriter(shape, bbox, crs, 'test', 1, outputdir, band_tags=band_tags)
    assert writer.outbands == 1
    assert os.path.basename(writer.file_names[0]) == 'test_band1.tif'
    mock_raster.update_tags.assert_called_once_with(1, image_type='Band1')


@patch('uncoverml.geoio.mpiops')
@patch('uncoverml.geoio.rasterio.open')
def test_imagewriter_write_and_close(mock_rasterio_open, mock_mpiops):
    mock_mpiops.chunk_index = 0
    mock_mpiops.chunks = 1
    mock_mpiops.comm.bcast = lambda x, root=0: x
    mock_mpiops.comm.barrier = MagicMock()
    mock_raster = MagicMock()
    mock_rasterio_open.return_value = mock_raster
    shape = (4, 4)
    bbox = np.array([[0, 0], [1, 1]])
    crs = 'EPSG:4326'
    band_tags = ['Band1']
    outputdir = tempfile.mkdtemp()
    writer = ImageWriter(shape, bbox, crs, 'test', 1, outputdir, band_tags=band_tags)
    data = np.ma.masked_array(
        data=np.random.rand(4 * 4, 1).astype(np.float32),
        mask=np.zeros((4 * 4, 1), dtype=bool)
    )
    writer.write(data, subchunk_index=0)
    writer.close()
    assert mock_raster.write.call_count == 1
    mock_raster.close.assert_called_once()

@patch('uncoverml.geoio.resample')
@patch('uncoverml.geoio.mpiops')
def test_output_thumbnails(mock_mpiops, mock_resample):
    mock_mpiops.chunk_index = 0
    mock_mpiops.chunks = 1
    dummy_file = '/tmp/fake_output_band1.tif'
    writer = ImageWriter.__new__(ImageWriter)
    writer.file_names = [dummy_file]
    writer.output_thumbnails(ratio=20)
    mock_resample.assert_called_with(dummy_file, output_tif='/tmp/fake_output_band1_thumbnail.tif', ratio=20)


def test_feature_names():
    config = SimpleNamespace()
    config.feature_sets = [
        SimpleNamespace(files=['/some/path/file3.tif', '/some/path/file1.tif']),
        SimpleNamespace(files=['/some/path/file2.tif'])
    ]
    result = geoio.feature_names(config)
    assert result == ['file1.tif', 'file3.tif', 'file2.tif']


@patch('uncoverml.geoio.log_missing_percentage')
@patch('uncoverml.geoio.RasterioImageSource')
def test_iterate_sources(mock_raster_src, mock_log_missing):
    f = lambda src: 'dummy_result'
    config = SimpleNamespace()
    config.is_prediction = False
    config.feature_sets = [SimpleNamespace(files=['/fake/file1.tif', '/fake/file2.tif'])]

    mock_raster_src.return_value = MagicMock()

    result = geoio._iterate_sources(f, config)
    assert isinstance(result, list)
    assert isinstance(result[0], OrderedDict)
    assert list(result[0].values()) == ['dummy_result', 'dummy_result']


@patch('uncoverml.geoio.mpiops')
@patch('uncoverml.geoio.log')
def test_log_missing_percentage(mock_log, mock_mpiops):
    mock_mpiops.count.return_value = 100
    mock_mpiops.comm.allreduce.return_value = 20
    mock_mpiops.chunks = 2
    x = np.ma.masked_array(data=np.ones((10, 10)), mask=np.zeros((10, 10)))
    geoio.log_missing_percentage('dummy.tif', x)
    mock_log.info.assert_called()


@patch('uncoverml.geoio._iterate_sources')
def test_image_resolutions(mock_iter):
    mock_iter.return_value = ['res']
    result = geoio.image_resolutions(SimpleNamespace())
    assert result == ['res']


@patch('uncoverml.geoio.features.extract_subchunks')
@patch('uncoverml.geoio._iterate_sources')
def test_image_subchunks(mock_iter, mock_extract):
    mock_extract.return_value = 'chunks'
    mock_iter.side_effect = lambda f, conf: [f(MagicMock())]
    conf = SimpleNamespace(is_prediction=False, prediction_template=None,
                           n_subchunks=1, patchsize=3)
    assert geoio.image_subchunks(0, conf) == ['chunks']


def test_extract_intersected_features():
    targets = MagicMock()
    targets.fields = {'img.tif': np.array([1, 2, 3])}
    config = SimpleNamespace(intersected_features={'img.tif': 'img.tif'})
    img = MagicMock()
    img.filename = '/path/img.tif'
    result = geoio.extract_intersected_features(img, targets, config)
    assert result.shape == (3, 1, 1, 1)
