import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from uncoverml import resampling


# Fixture for creating a dummy GeoDataFrame
@pytest.fixture
def dummy_gdf():
    np.random.seed(42)
    size = 100
    df = pd.DataFrame({
        'target': np.random.rand(size) * 100,
        'keep_field': np.random.randint(0, 5, size)
    })
    df['geometry'] = [Point(x, y) for x, y in zip(np.random.rand(size)*10, np.random.rand(size)*10)]
    return gpd.GeoDataFrame(df, geometry='geometry')


def test_bootstrap_data_indices():
    indices = resampling.bootstrap_data_indicies(10, samples=5, random_state=42)
    assert len(indices) == 5
    assert indices.max() < 10
    assert indices.min() >= 0


def test_prepapre_dataframe_fields(dummy_gdf, monkeypatch):
    def mock_read_file(path):
        return dummy_gdf

    monkeypatch.setattr(gpd, "read_file", mock_read_file)

    out = resampling.prepapre_dataframe("fake_path.shp", target_field="target", fields_to_keep=["keep_field"])
    assert "keep_field" in out.columns
    assert "target" in out.columns
    assert "geometry" in out.columns


def test_filter_fields_valid(dummy_gdf):
    fields = ["keep_field", "target"]
    out = resampling.filter_fields(fields, dummy_gdf)
    for field in fields + ['geometry']:
        assert field in out.columns


def test_filter_fields_invalid(dummy_gdf):
    with pytest.raises(RuntimeError):
        resampling.filter_fields(["nonexistent_field"], dummy_gdf)


def test_resample_by_magnitude_linear(dummy_gdf, monkeypatch):
    monkeypatch.setattr(gpd, "read_file", lambda path: dummy_gdf)
    out = resampling.resample_by_magnitude(
        "fake_path.shp",
        target_field="target",
        bins=5,
        interval='linear',
        fields_to_keep=["keep_field"],
        bootstrap=True,
        output_samples=50
    )
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.shape[0] >= 50


def test_resample_by_magnitude_percentile(dummy_gdf, monkeypatch):
    monkeypatch.setattr(gpd, "read_file", lambda path: dummy_gdf)
    out = resampling.resample_by_magnitude(
        "fake_path.shp",
        target_field="target",
        bins=5,
        interval='percentile',
        fields_to_keep=["keep_field"],
        bootstrap=True,
        output_samples=60
    )
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.shape[0] >= 50


def test_resample_spatially(dummy_gdf, monkeypatch):
    monkeypatch.setattr(gpd, "read_file", lambda path: dummy_gdf)
    out = resampling.resample_spatially(
        "dummy_path.shp",
        target_field="target",
        rows=5,
        cols=5,
        bootstrap=True,
        fields_to_keep=["keep_field"],
        output_samples=50
    )
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.shape[0] >= 50


def test_create_grouping_polygons(dummy_gdf):
    polys = resampling.create_grouping_polygons_from_geo_df(4, 4, dummy_gdf)
    assert len(polys) == 16
    for poly in polys:
        assert poly.is_valid


def test_sample_without_replacement_enough_samples(dummy_gdf):
    df = dummy_gdf.sample(n=20)
    main, val = resampling._sample_without_replacement(df, 10, validate=True)
    assert len(main) == 10
    assert isinstance(val, gpd.GeoDataFrame)

def test_sample_without_replacement_too_few_samples(dummy_gdf):
    df = dummy_gdf.sample(n=5)
    main, val = resampling._sample_without_replacement(df, 10, validate=True)
    assert main.equals(df)
    assert val.empty
