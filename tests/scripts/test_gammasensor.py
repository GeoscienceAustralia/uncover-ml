import os
import sys
import glob
import numpy as np
import pytest
import click
from click.testing import CliRunner

from uncoverml.scripts import gammasensor


class DummyImageSource:
    def __init__(self, filepath):
        self.filepath = filepath


class DummyImage:
    def __init__(self, image_source):
        self.resolution = (2, 2, 1)
        self.pixsize_x = 1.0
        self.pixsize_y = 1.0
        arr = np.arange(4).reshape((2, 2, 1)).astype(float)
        self._data = np.ma.MaskedArray(data=arr, mask=np.zeros_like(arr, dtype=bool))
        self.crs = "EPSG:4326"

    def data(self):
        return self._data

    def patched_shape(self, idx):
        return (self.resolution[0], self.resolution[1])

    def patched_bbox(self, idx):
        return ((0, 0), (2, 2))


class DummyWriter:
    instances = []

    def __init__(self, shape, bbox, crs, name, n_subchunks, outputdir, band_tags, independent):
        self.shape = shape
        self.bbox = bbox
        self.crs = crs
        self.name = name
        self.n_subchunks = n_subchunks
        self.outputdir = outputdir
        self.band_tags = band_tags
        self.independent = independent
        self.written = []  # store (data, idx)
        DummyWriter.instances.append(self)

    def write(self, data, idx):
        self.written.append((data.copy(), idx))


@pytest.fixture(autouse=True)
def patch_geoio_and_image(monkeypatch, tmp_path):
    monkeypatch.setattr(gammasensor.geoio, "RasterioImageSource", DummyImageSource)
    monkeypatch.setattr(gammasensor, "Image", DummyImage)

    def dummy_sensor_footprint(w, h, px, py, height, absorption):
        return np.array([[1.0]])

    monkeypatch.setattr(gammasensor.filtering, "sensor_footprint", dummy_sensor_footprint)
    monkeypatch.setattr(gammasensor.filtering, "fwd_filter", lambda data, S: data)
    monkeypatch.setattr(gammasensor.filtering, "inv_filter", lambda data, S, noise: data)
    monkeypatch.setattr(gammasensor.filtering, "kernel_impute", lambda data, S: data)

    monkeypatch.setattr(gammasensor.mpiops, "chunks", 1)
    monkeypatch.setattr(gammasensor.mpiops, "chunk_index", 0)

    monkeypatch.setattr(gammasensor.geoio, "ImageWriter", DummyWriter)

    monkeypatch.chdir(tmp_path)
    yield
    DummyWriter.instances.clear()


def test_write_data_single_band(tmp_path):
    image_source = DummyImageSource("dummy.tif")
    image = DummyImage(image_source)
    arr = np.arange(4).reshape((2, 2, 1)).astype(float)
    masked_arr = np.ma.MaskedArray(data=arr, mask=np.zeros_like(arr, dtype=bool))

    gammasensor.write_data(
        data=masked_arr,
        name="testimg",
        in_image=image,
        outputdir=str(tmp_path),
        forward=True
    )

    assert len(DummyWriter.instances) == 1
    writer = DummyWriter.instances[0]

    assert writer.shape == (2, 2, 1)
    assert writer.bbox == ((0, 0), (2, 2))
    assert writer.crs == "EPSG:4326"
    assert writer.name == "testimg"
    assert writer.n_subchunks == 1
    assert writer.outputdir == str(tmp_path)
    assert writer.band_tags == ["convolved"]
    assert writer.independent is True

    assert len(writer.written) == 1
    written_data, idx = writer.written[0]
    assert written_data.shape == (4, 1)
    np.testing.assert_array_equal(written_data, arr.reshape(-1, 1))
    assert idx == 0


def test_write_data_multi_band(tmp_path):
    image_source = DummyImageSource("dummy.tif")
    image = DummyImage(image_source)
    image.resolution = (2, 2, 2)
    arr = np.stack([
        np.full((2, 2), fill_value=5.0),
        np.full((2, 2), fill_value=7.0)
    ], axis=2)
    masked_arr = np.ma.MaskedArray(data=arr, mask=np.zeros_like(arr, dtype=bool))
    gammasensor.write_data(
        data=masked_arr,
        name="multiband",
        in_image=image,
        outputdir=str(tmp_path),
        forward=False
    )

    assert len(DummyWriter.instances) == 1
    writer = DummyWriter.instances[0]

    assert writer.band_tags == ["deconvolved_band1", "deconvolved_band2"]
    assert writer.shape == (2, 2, 2)

    assert len(writer.written) == 1
    written_data, idx = writer.written[0]
    assert written_data.shape == (4, 2)
    expected = arr.reshape(-1, 2)
    np.testing.assert_array_equal(written_data, expected)
    assert idx == 0


def test_cli_no_files_found(monkeypatch):
    monkeypatch.setattr(gammasensor.os.path, "isdir", lambda p: False)
    monkeypatch.setattr(gammasensor.glob, "glob", lambda pattern: [])

    runner = CliRunner()
    result = runner.invoke(
        gammasensor.cli,
        [
            "--height", "10.0",
            "--absorption", "0.1",
            "nonexistent_pattern.tif",
        ],
        catch_exceptions=True
    )

    assert result.exit_code == 0
    assert "No files found. Exiting" in result.output


def test_cli_forward_process(tmp_path, monkeypatch):
    dummy_tif = str(tmp_path / "dummy.tif")
    open(dummy_tif, "a").close()
    monkeypatch.setattr(gammasensor.os.path, "isdir", lambda p: False)
    monkeypatch.setattr(gammasensor.glob, "glob", lambda pattern: [dummy_tif])
    called = {"count": 0}
    def fake_write_data(data, name, in_image, outputdir, forward):
        called["count"] += 1
        assert forward is True
        assert name == "dummy"
        assert outputdir == str(tmp_path)
        assert isinstance(data, np.ma.MaskedArray)
    monkeypatch.setattr(gammasensor, "write_data", fake_write_data)

    runner = CliRunner()
    result = runner.invoke(
        gammasensor.cli,
        [
            "--height", "10.0",
            "--absorption", "0.1",
            "--apply",
            "--outputdir", str(tmp_path),
            dummy_tif,
        ],
        catch_exceptions=True
    )
    assert result.exit_code == 0
    assert called["count"] == 1


def test_cli_backward_with_impute(monkeypatch):
    tmp_dir = os.getcwd()
    dummy_tif = os.path.join(tmp_dir, "dummy.tif")
    open(dummy_tif, "a").close()
    monkeypatch.setattr(gammasensor.os.path, "isdir", lambda p: False)
    monkeypatch.setattr(gammasensor.glob, "glob", lambda pattern: [dummy_tif])
    impute_called = {"count": 0}

    def fake_kernel_impute(data, S):
        impute_called["count"] += 1
        return data

    monkeypatch.setattr(gammasensor.filtering, "kernel_impute", fake_kernel_impute)
    called = {"forward_flags": []}
    def fake_write_data(data, name, in_image, outputdir, forward):
        called["forward_flags"].append(forward)
        assert forward is False
    monkeypatch.setattr(gammasensor, "write_data", fake_write_data)

    runner = CliRunner()
    result = runner.invoke(
        gammasensor.cli,
        [
            "--height", "10.0",
            "--absorption", "0.1",
            "--invert",
            "--impute",
            dummy_tif,
        ],
        catch_exceptions=True
    )
    assert result.exit_code == 0
    assert impute_called["count"] == 0
    assert len(called["forward_flags"]) == 1
    assert called["forward_flags"][0] is False
