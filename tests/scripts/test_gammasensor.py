import os
import sys
import glob
import numpy as np
import pytest
import click
from click.testing import CliRunner

from uncoverml.scripts import gammasensor


# --------------------
# Fixtures and Helpers
# --------------------

class DummyImageSource:
    """Dummy replacement for geoio.RasterioImageSource."""
    def __init__(self, filepath):
        self.filepath = filepath


class DummyImage:
    """Dummy replacement for uncoverml.image.Image."""
    def __init__(self, image_source):
        # Simulate resolution: (width_pixels, height_pixels, channels)
        self.resolution = (2, 2, 1)
        # Pixel sizes for sensor_footprint
        self.pixsize_x = 1.0
        self.pixsize_y = 1.0
        # Create a 2×2×1 masked array with no masked entries
        arr = np.arange(4).reshape((2, 2, 1)).astype(float)
        self._data = np.ma.MaskedArray(data=arr, mask=np.zeros_like(arr, dtype=bool))
        # Dummy CRS
        self.crs = "EPSG:4326"

    def data(self):
        return self._data

    def patched_shape(self, idx):
        # Return (width, height) without the channel dimension
        return (self.resolution[0], self.resolution[1])

    def patched_bbox(self, idx):
        # Dummy bounding box: ((xmin, ymin), (xmax, ymax))
        return ((0, 0), (2, 2))


class DummyWriter:
    """Dummy replacement for geoio.ImageWriter to capture calls."""
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
    """
    Before each test:
      - Replace geoio.RasterioImageSource → DummyImageSource
      - Replace gammasensor.Image → DummyImage
      - Stub out filtering.* to no-op or identity
      - Force mpiops.chunks=1, mpiops.chunk_index=0
      - Replace geoio.ImageWriter → DummyWriter
      - Change working dir to tmp_path
    """
    # 1. Patch RasterioImageSource → DummyImageSource
    monkeypatch.setattr(gammasensor.geoio, "RasterioImageSource", DummyImageSource)

    # 2. Patch Image → DummyImage
    monkeypatch.setattr(gammasensor, "Image", DummyImage)

    # 3. Stub filtering.sensor_footprint → 1×1 identity kernel
    def dummy_sensor_footprint(w, h, px, py, height, absorption):
        return np.array([[1.0]])
    monkeypatch.setattr(gammasensor.filtering, "sensor_footprint", dummy_sensor_footprint)

    # 4. Stub fwd_filter, inv_filter, kernel_impute → return data unchanged
    monkeypatch.setattr(gammasensor.filtering, "fwd_filter", lambda data, S: data)
    monkeypatch.setattr(gammasensor.filtering, "inv_filter", lambda data, S, noise: data)
    monkeypatch.setattr(gammasensor.filtering, "kernel_impute", lambda data, S: data)

    # 5. Force MPI to be single-chunk
    monkeypatch.setattr(gammasensor.mpiops, "chunks", 1)
    monkeypatch.setattr(gammasensor.mpiops, "chunk_index", 0)

    # 6. Replace ImageWriter → DummyWriter
    monkeypatch.setattr(gammasensor.geoio, "ImageWriter", DummyWriter)

    # 7. Change cwd to a temporary directory
    monkeypatch.chdir(tmp_path)

    yield

    # Cleanup DummyWriter instances
    DummyWriter.instances.clear()


# --------------------
# Tests for write_data
# --------------------

def test_write_data_single_band(tmp_path):
    """
    If the input array has shape (2,2,1) and forward=True,
    DummyWriter should receive shape=(2,2,1), band_tags=["convolved"],
    and write() should get a (4,1) array.
    """
    # 1. Create a DummyImage with resolution (2,2,1)
    image_source = DummyImageSource("dummy.tif")
    image = DummyImage(image_source)

    # 2. Build a 2×2×1 masked array
    arr = np.arange(4).reshape((2, 2, 1)).astype(float)
    masked_arr = np.ma.MaskedArray(data=arr, mask=np.zeros_like(arr, dtype=bool))

    # 3. Call write_data(...)
    gammasensor.write_data(
        data=masked_arr,
        name="testimg",
        in_image=image,
        outputdir=str(tmp_path),
        forward=True
    )

    # There should be exactly one DummyWriter instance
    assert len(DummyWriter.instances) == 1
    writer = DummyWriter.instances[0]

    # Check constructor arguments
    # patched_shape → (2,2); channels = 1 → shape = (2,2,1)
    assert writer.shape == (2, 2, 1)
    assert writer.bbox == ((0, 0), (2, 2))
    assert writer.crs == "EPSG:4326"
    assert writer.name == "testimg"
    assert writer.n_subchunks == 1
    assert writer.outputdir == str(tmp_path)
    # Single-band and forward=True → band_tags = ["convolved"]
    assert writer.band_tags == ["convolved"]
    assert writer.independent is True

    # Verify that write() was called once
    assert len(writer.written) == 1
    written_data, idx = writer.written[0]
    # The data passed should be flattened: (2*2) × 1 → (4,1)
    assert written_data.shape == (4, 1)
    np.testing.assert_array_equal(written_data, arr.reshape(-1, 1))
    assert idx == 0


def test_write_data_multi_band(tmp_path):
    """
    If the input array has shape (2,2,2) and forward=False,
    DummyWriter should receive shape=(2,2,2), band_tags=["deconvolved_band1","deconvolved_band2"],
    and write() should get a (4,2) array.
    """
    # 1. Create a DummyImage with resolution (2,2,1) but override to 2 channels
    image_source = DummyImageSource("dummy.tif")
    image = DummyImage(image_source)
    image.resolution = (2, 2, 2)

    # 2. Build a 2×2×2 array and mask = all False
    arr = np.stack([
        np.full((2, 2), fill_value=5.0),
        np.full((2, 2), fill_value=7.0)
    ], axis=2)
    masked_arr = np.ma.MaskedArray(data=arr, mask=np.zeros_like(arr, dtype=bool))

    # 3. Call write_data(...) in backward (forward=False) mode
    gammasensor.write_data(
        data=masked_arr,
        name="multiband",
        in_image=image,
        outputdir=str(tmp_path),
        forward=False
    )

    # Exactly one DummyWriter instance
    assert len(DummyWriter.instances) == 1
    writer = DummyWriter.instances[0]

    # Two bands → tags = ["deconvolved_band1", "deconvolved_band2"]
    assert writer.band_tags == ["deconvolved_band1", "deconvolved_band2"]
    # patched_shape=(2,2), channels=2 → shape=(2,2,2)
    assert writer.shape == (2, 2, 2)

    # Check that write() was called once with flattened shape (4,2)
    assert len(writer.written) == 1
    written_data, idx = writer.written[0]
    assert written_data.shape == (4, 2)
    expected = arr.reshape(-1, 2)
    np.testing.assert_array_equal(written_data, expected)
    assert idx == 0


# --------------------
# Tests for the CLI
# --------------------

def test_cli_no_files_found(monkeypatch):
    """
    If glob.glob([]) yields no files, the CLI calls sys.exit() without an argument,
    resulting in exit_code == 0. We still verify that "No files found. Exiting"
    appears in the output.
    """
    # 1. Patch os.path.isdir so it never treats the argument as a directory
    monkeypatch.setattr(gammasensor.os.path, "isdir", lambda p: False)

    # 2. Patch glob.glob to always return an empty list
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

    # sys.exit() without an argument produces exit_code == 0
    assert result.exit_code == 0
    assert "No files found. Exiting" in result.output


def test_cli_forward_process(tmp_path, monkeypatch):
    """
    Test the CLI in forward mode. Monkey-patch file discovery and write_data.
    Ensure write_data is called exactly once with forward=True.
    """
    # 1. Create a dummy .tif path
    dummy_tif = str(tmp_path / "dummy.tif")
    open(dummy_tif, "a").close()  # create an empty file

    # 2. Patch os.path.isdir → False, so CLI treats it as a file pattern
    monkeypatch.setattr(gammasensor.os.path, "isdir", lambda p: False)

    # 3. Patch glob.glob to return exactly our dummy file
    monkeypatch.setattr(gammasensor.glob, "glob", lambda pattern: [dummy_tif])

    # 4. Patch write_data to capture the calls
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
            "--apply",               # forward mode
            "--outputdir", str(tmp_path),
            dummy_tif,
        ],
        catch_exceptions=True
    )
    assert result.exit_code == 0
    assert called["count"] == 1


def test_cli_backward_with_impute(monkeypatch):
    """
    Test the CLI in backward (invert) mode with --impute. In the default dummy setup,
    the data has no masked entries, so filtering.kernel_impute is NOT called.
    We assert that impute_called["count"] == 0 in that scenario.
    Also verify write_data is still called once.
    """
    # 1. Create a dummy .tif path in the current (temp) directory
    tmp_dir = os.getcwd()  # patch_geoio_and_image has already changed cwd to a tempdir
    dummy_tif = os.path.join(tmp_dir, "dummy.tif")
    open(dummy_tif, "a").close()

    # 2. Patch os.path.isdir → False
    monkeypatch.setattr(gammasensor.os.path, "isdir", lambda p: False)

    # 3. Patch glob.glob → [dummy_tif]
    monkeypatch.setattr(gammasensor.glob, "glob", lambda pattern: [dummy_tif])

    # 4. Track kernel_impute calls
    impute_called = {"count": 0}
    def fake_kernel_impute(data, S):
        impute_called["count"] += 1
        return data
    monkeypatch.setattr(gammasensor.filtering, "kernel_impute", fake_kernel_impute)

    # 5. Track write_data forward flags
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
            "--invert",      # backward mode
            "--impute",      # request imputation
            dummy_tif,
        ],
        catch_exceptions=True
    )
    assert result.exit_code == 0

    # Because the DummyImage has no masked entries, kernel_impute should NOT be called
    assert impute_called["count"] == 0

    # write_data must still be called exactly once in backward mode
    assert len(called["forward_flags"]) == 1
    assert called["forward_flags"][0] is False
