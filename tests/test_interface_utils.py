import io
import os
import json
import shutil
import zipfile
import tempfile
import builtins
import pytest
import numpy as np
import rasterio
from unittest import mock
from pathlib import Path
from uncoverml.interface_utils import (
    rename_files_before_upload,
    calc_std,
    calc_uncert,
    stretch_raster,
    create_thumbnail,
    create_results_zip,
    read_presigned_urls_and_upload,
)


class DummyConfig:
    def __init__(self, output_dir):
        self.output_dir = output_dir


@pytest.fixture
def tmp_output_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def create_dummy_tif(file_path, data, profile=None):
    if profile is None:
        profile = {
            'driver': 'GTiff',
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'dtype': str(data.dtype),
            'crs': None,
            'transform': rasterio.Affine(1, 0, 0, 0, -1, 0)
        }
    with rasterio.open(file_path, 'w', **profile) as dst:
        dst.write(data, 1)


def test_rename_files_before_upload(tmp_output_dir):
    (tmp_output_dir / 'transformedrandomforest_testfile.txt').write_text("test")
    (tmp_output_dir / 'config.model').write_text("model")

    config = DummyConfig(tmp_output_dir)
    rename_files_before_upload(config)

    assert not (tmp_output_dir / 'transformedrandomforest_testfile.txt').exists()
    assert (tmp_output_dir / 'testfile.txt').exists()
    assert (tmp_output_dir / 'config.model').exists()


def test_calc_std_creates_std_tif(tmp_output_dir):
    var_data = np.array([[4, 9], [16, 25]], dtype=np.float32)
    create_dummy_tif(tmp_output_dir / 'variance.tif', var_data)

    config = DummyConfig(tmp_output_dir)
    calc_std(config)

    std_file = tmp_output_dir / 'std.tif'
    assert std_file.exists()

    with rasterio.open(std_file) as src:
        data = src.read(1)
    np.testing.assert_almost_equal(data, np.sqrt(var_data))


def test_calc_uncert(tmp_output_dir):
    pred_data = np.array([[2, 4], [6, 8]], dtype=np.float32)
    var_data = np.array([[1, 4], [9, 16]], dtype=np.float32)

    create_dummy_tif(tmp_output_dir / 'prediction.tif', pred_data)
    create_dummy_tif(tmp_output_dir / 'variance.tif', var_data)

    config = DummyConfig(tmp_output_dir)
    calc_uncert(config)

    uncert_file = tmp_output_dir / 'uncert.tif'
    assert uncert_file.exists()

    with rasterio.open(uncert_file) as src:
        out_data = src.read(1)
    expected = (np.sqrt(var_data)) / pred_data
    np.testing.assert_almost_equal(out_data, expected)


def test_stretch_raster():
    data = np.array([0, 1, 2, 3, 4, 5, np.nan])
    stretched = stretch_raster(data, pct_lims=[0, 100])
    assert stretched.min() >= 0
    assert stretched.max() <= 255
    assert stretched[-1] == 0


@mock.patch('matplotlib.pyplot.subplots')
def test_create_thumbnail(mock_subplots, tmp_path):
    class DummyConfig:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    config = DummyConfig(tmp_path)

    mock_src = mock.Mock()
    mock_src.read.return_value = np.array([[1, 2], [3, 4]], dtype=np.float32)
    mock_src.nodata = None

    with mock.patch('rasterio.open', autospec=True) as mock_rasterio_open:
        mock_rasterio_open.return_value.__enter__.return_value = mock_src
        mock_rasterio_open.return_value.__exit__.return_value = None
        mock_fig = mock.Mock()
        mock_ax = mock.Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        create_thumbnail(config, 'test')
        mock_fig.savefig.assert_called_once()
        save_path = mock_fig.savefig.call_args[0][0]
        assert 'test_thumbnail.png' in str(save_path)


def test_create_results_zip(tmp_output_dir):
    for i in range(3):
        (tmp_output_dir / f'file{i}.txt').write_text(f"content {i}")

    config = DummyConfig(tmp_output_dir)
    create_results_zip(config)

    zip_file = tmp_output_dir / 'full_results.zip'
    assert zip_file.exists()

    with zipfile.ZipFile(zip_file) as z:
        names = z.namelist()
        for i in range(3):
            assert f'file{i}.txt' in names


@mock.patch('requests.post')
def test_read_presigned_urls_and_upload(mock_post, tmp_output_dir):
    parent_dir = tmp_output_dir.parent
    json_data = [
        {
            'url': 'http://example.com/upload',
            'fields': {'key': 'somefolder/file1.txt'}
        },
        {
            'url': 'http://example.com/upload',
            'fields': {'key': 'somefolder/file2.txt'}
        }
    ]
    json_file = parent_dir / 'upload_urls.json'
    json_file.write_text(json.dumps(json_data))

    for entry in json_data:
        file_name = entry['fields']['key'].split('/')[-1]
        (tmp_output_dir / file_name).write_text(f"dummy content for {file_name}")

    mock_resp = mock.Mock()
    mock_resp.status_code = 204
    mock_post.return_value = mock_resp

    config = DummyConfig(tmp_output_dir)
    read_presigned_urls_and_upload(config, job_type='other')

    assert mock_post.call_count == 2
    called_files = []
    for call in mock_post.call_args_list:
        files_dict = call[1]['files']
        called_files.append(files_dict['file'][0])

    for entry in json_data:
        expected_file = tmp_output_dir / entry['fields']['key'].split('/')[-1]
        assert str(expected_file) in called_files
