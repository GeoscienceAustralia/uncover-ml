import os
import shutil
import pytest
import shapefile
from click.testing import CliRunner

from uncoverml.scripts import subsampletargets as script


@pytest.fixture
def tmp_shapefile(tmp_path):
    shp_prefix = tmp_path / "input"
    writer = shapefile.Writer(str(shp_prefix), shapeType=shapefile.POINT)
    writer.field("id", "N")
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    for i, (x, y) in enumerate(points):
        writer.point(x, y)
        writer.record(i)
    writer.close()
    return str(shp_prefix)


def test_subsample_npoints_exceeds(tmp_shapefile, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        script.cli,
        [
            tmp_shapefile + ".shp",
            "-n", "10",
            "-o", str(tmp_path),
            "-v", "INFO"
        ],
        catch_exceptions=True
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    msg = str(result.exception)
    assert "Sample larger than population" in msg
