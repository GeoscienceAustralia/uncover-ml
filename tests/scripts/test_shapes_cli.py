import os
import pytest
import click
from click.testing import CliRunner
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon

from uncoverml.scripts import shapes_cli as script


class DummyConfigGroup:
    def __init__(self, target_file, grouped_output, spatial_grouping_args, output_group_col_name):
        self.target_file = target_file
        self.grouped_output = grouped_output
        self.spatial_grouping_args = spatial_grouping_args
        self.output_group_col_name = output_group_col_name


class DummyConfigSplit:
    def __init__(self, grouped_output, split_group_col_name, split_oos_fraction, train_shapefile, oos_shapefile):
        self.grouped_output = grouped_output
        self.split_group_col_name = split_group_col_name
        self.split_oos_fraction = split_oos_fraction
        self.train_shapefile = train_shapefile
        self.oos_shapefile = oos_shapefile


@pytest.fixture(autouse=True)
def patch_config_and_io(monkeypatch, tmp_path):
    monkeypatch.setattr(script.ls.config, "Config", lambda path: None)
    monkeypatch.setattr(
        script.ls.resampling,
        "create_grouping_polygons_from_geo_df",
        lambda rows, cols, gdf: []
    )

    saved = {"grouped": None, "grouped_path": None, "split_train": None, "split_oos": None}
    original_to_file = gpd.GeoDataFrame.to_file

    def fake_to_file(self, path, *args, **kwargs):
        if "grouped" in path:
            saved["grouped"] = self.copy()
            saved["grouped_path"] = path
        elif "train" in path:
            saved["split_train"] = self.copy()
        elif "oos" in path:
            saved["split_oos"] = self.copy()
        open(path, "w").close()

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)

    yield saved

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", original_to_file)


def test_cli_group_creates_grouped_gdf(patch_config_and_io, monkeypatch, tmp_path):
    geom = [Point(0, 0), Point(10, 10)]
    gdf = gpd.GeoDataFrame({"value": [1, 2]}, geometry=geom)

    monkeypatch.setattr(gpd, "read_file", lambda path: gdf.copy())

    poly0 = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    poly1 = Polygon([(9, 9), (11, 9), (11, 11), (9, 11)])
    monkeypatch.setattr(
        script.ls.resampling,
        "create_grouping_polygons_from_geo_df",
        lambda rows, cols, gdf_arg: [poly0, poly1]
    )

    cfg = DummyConfigGroup(
        target_file="ignored_input.shp",
        grouped_output=str(tmp_path / "grouped_output.shp"),
        spatial_grouping_args={"rows": 1, "cols": 2},
        output_group_col_name="group_id"
    )
    monkeypatch.setattr(script.ls.config, "Config", lambda path: cfg)

    runner = CliRunner()
    result = runner.invoke(
        script.cli,
        ["group", "pipeline.yaml", "--verbosity", "INFO"],
        catch_exceptions=False
    )
    assert result.exit_code == 0

    saved = patch_config_and_io
    assert saved["grouped_path"] == cfg.grouped_output

    grouped_gdf = saved["grouped"]
    assert "group_id" in grouped_gdf.columns
    vals = sorted(grouped_gdf["group_id"].tolist())
    assert vals == [0, 1]


def test_cli_split_creates_train_and_oos(patch_config_and_io, monkeypatch, tmp_path):
    geom = [Point(0, 0), Point(1, 1), Point(2, 2)]
    groups = [0, 1, 2]
    gdf = gpd.GeoDataFrame({"group": groups}, geometry=geom)
    monkeypatch.setattr(gpd, "read_file", lambda path: gdf.copy())
