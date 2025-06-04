import os
import sys
import zipfile
import types
import pytest
from click.testing import CliRunner
from PIL import Image as PILImage

simplekml_mod = types.ModuleType("simplekml")

class DummyGround:
    def __init__(self, name):
        self.name = name
        self.icon = types.SimpleNamespace(href=None)
        self.latlonbox = types.SimpleNamespace(west=None, east=None, north=None, south=None)

class DummyKml:
    def __init__(self):
        self.overlays = []
    def newgroundoverlay(self, name=None):
        ground = DummyGround(name)
        self.overlays.append(ground)
        return ground
    def savekmz(self, path):
        base = os.path.splitext(path)[0]
        jpg_file = f"{base}.jpg"
        overlay_name = self.overlays[0].name if self.overlays else ""
        kml_content = f"<kml><Document><name>{overlay_name}</name></Document></kml>"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("doc.kml", kml_content)
            if os.path.exists(jpg_file):
                z.write(jpg_file, arcname=os.path.basename(jpg_file))

simplekml_mod.Kml = DummyKml
sys.modules["simplekml"] = simplekml_mod

simplekml_mod.Kml = DummyKml
sys.modules["simplekml"] = simplekml_mod

mpi4py_mod = types.ModuleType("mpi4py")
mpi4py_mod.MPI = types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(Get_rank=lambda: 0))

sys.modules["mpi4py"] = mpi4py_mod
sys.modules["mpi4py.MPI"] = mpi4py_mod.MPI

from uncoverml.scripts.tiff2kmz import main
import uncoverml.geoio as geoio_mod


class DummyImageObj:
    def __init__(self):
        self.xmin = -10.0
        self.xmax = 10.0
        self.ymin = -5.0
        self.ymax = 5.0


@pytest.fixture(autouse=True)
def patch_geoio(monkeypatch):
    monkeypatch.setattr(geoio_mod, "Image", lambda path: DummyImageObj(), raising=False)
    yield


def create_dummy_tiff(path):
    img = PILImage.new("RGB", (10, 10), color=(123, 222, 111))
    img.save(path, format="TIFF")


def test_default_outfile_and_kmz(tmp_path):
    tiff_path = tmp_path / "test_input.tif"
    create_dummy_tiff(str(tiff_path))

    runner = CliRunner()
    result = runner.invoke(main, [str(tiff_path)], catch_exceptions=False)
    assert result.exit_code == 0

    base = tiff_path.stem
    jpg_path = tmp_path / f"{base}.jpg"
    kmz_path = tmp_path / f"{base}.kmz"
    assert jpg_path.exists()
    assert kmz_path.exists()

    with zipfile.ZipFile(str(kmz_path), "r") as z:
        namelist = z.namelist()
        assert "doc.kml" in namelist
        assert f"{base}.jpg" in namelist
        kml_text = z.read("doc.kml").decode("utf-8")
        assert f"<name>{base}</name>" in kml_text


def test_custom_outfile_and_overlayname(tmp_path):
    tiff_path = tmp_path / "my_input.tif"
    create_dummy_tiff(str(tiff_path))

    custom_out = tmp_path / "custom_name"
    overlay = "MyOverlay"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--outfile", str(custom_out), "--overlayname", overlay, str(tiff_path)],
        catch_exceptions=False
    )
    assert result.exit_code == 0

    jpg_path = tmp_path / "custom_name.jpg"
    kmz_path = tmp_path / "custom_name.kmz"
    assert jpg_path.exists()
    assert kmz_path.exists()

    with zipfile.ZipFile(str(kmz_path), "r") as z:
        namelist = z.namelist()
        assert "doc.kml" in namelist
        assert "custom_name.jpg" in namelist
        kml_text = z.read("doc.kml").decode("utf-8")
        assert f"<name>{overlay}</name>" in kml_text
