# tests/test_shap_module.py

import logging
import pytest
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from collections import OrderedDict
import uncoverml.shapley as sm
from unittest.mock import patch, MagicMock, mock_open
from shapely.geometry import Point, Polygon

@patch('uncoverml.shapley.intersect_shp')
def test_get_data_points(mock_intersect_shp):
    points = [Point(1, 1), Point(2, 2)]
    gdf = gpd.GeoDataFrame({'geometry': points}, geometry='geometry')
    mock_intersect_shp.side_effect = [
        (np.array([[1.0]]), None, (1, 1)),
        (np.array([[2.0]]), None, (1, 1)),
    ]
    image_path = '/fake/path/image.tif'
    result = sm.get_data_points(gdf, image_path)
    assert isinstance(result, np.ndarray)
    assert mock_intersect_shp.call_count == 2

@patch('uncoverml.shapley.intersect_shp')
def test_get_data_polygon(mock_intersect_shp):
    poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({'geometry': [poly]}, geometry='geometry')
    dummy_result = np.array([[1.0, 2.0], [3.0, 4.0]])
    dummy_lonlat = (np.array([1, 2]), np.array([3, 4]))
    dummy_shape = dummy_result.shape
    mock_intersect_shp.return_value = (dummy_result, dummy_lonlat, dummy_shape)
    image_path = '/fake/path/image.tif'
    result, lon_lat, shape = sm.get_data_polygon(gdf, image_path)
    np.testing.assert_array_equal(result, dummy_result)
    assert lon_lat == dummy_lonlat
    assert shape == dummy_shape
    mock_intersect_shp.assert_called_once_with(gdf, image_path, type='poly')

@patch('uncoverml.shapley.mpiops.count')
@patch('uncoverml.shapley.mpiops.comm')
@patch('uncoverml.shapley.get_data_polygon')
@patch('uncoverml.shapley.gpd.read_file')
@patch('uncoverml.shapley.missing_percentage')
def test_image_feature_sets_shap_polygon(mock_missing, mock_readfile, mock_get_polygon,
                                         mock_comm, mock_count):
    dummy_gdf = gpd.GeoDataFrame({'geometry': [MagicMock()]})
    mock_readfile.return_value = dummy_gdf
    mock_get_polygon.return_value = (np.array([[1, 2], [3, 4]]), [('x', 'y')], (2, 2))
    mock_count.return_value = 4
    mock_comm.allreduce.return_value = 20
    shap_config = MagicMock()
    shap_config.shapefile = {'type': 'poly', 'dir': '/some/path'}
    shap_config.feature_path = '/some'
    fs = MagicMock()
    fs.files = ['/some/feature1.tif']
    main_config = MagicMock()
    main_config.feature_sets = [fs]
    results, coords = sm.image_feature_sets_shap(shap_config, main_config)
    assert isinstance(results, list)
    assert isinstance(results[0], OrderedDict)
    assert '/some/feature1.tif' in results[0]
    assert coords['/feature1'] == ([('x', 'y')], (2, 2))


@patch('uncoverml.shapley.features.gather_features')
@patch('uncoverml.shapley.features.transform_features')
@patch('uncoverml.shapley.image_feature_sets_shap')
def test_load_data_shap_polygon(mock_image_sets, mock_transform, mock_gather):
    shap_config = MagicMock()
    shap_config.shapefile = {'type': 'poly'}
    feature_set = MagicMock()
    feature_set.transform_set = 'transform_set_1'
    main_config = MagicMock()
    main_config.feature_sets = [feature_set]
    main_config.final_transform = 'final_transform'
    dummy_image_chunks = [[{'/some/path.tif': np.array([[[1]]])}]]
    dummy_coords = {'feature1': (['x', 'y'], (1, 1))}
    mock_image_sets.return_value = (dummy_image_chunks, dummy_coords)
    dummy_features = {True: np.array([[1, 2, 3]])}
    mock_transform.return_value = (dummy_features, True)
    mock_gather.return_value = np.array([[1, 2, 3]])
    result, coords = sm.load_data_shap(shap_config, main_config)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)
    assert coords == dummy_coords


@patch('uncoverml.shapley.gen_poly_data')
@patch('uncoverml.shapley.gpd.read_file')
@patch('uncoverml.shapley.log.info')
def test_load_point_poly_data(mock_log, mock_read, mock_genpoly):
    gdf = gpd.GeoDataFrame({'Name': ['P1', 'P2'], 'geometry': [MagicMock(), MagicMock()]})
    mock_read.return_value = gdf
    mock_genpoly.side_effect = [
        (np.array([[1]]), np.array([[100.0, 200.0]])),
        (np.array([[2]]), np.array([[110.0, 210.0]]))
    ]
    shap_config = MagicMock()
    shap_config.shapefile = {'dir': '/some/path', 'type': 'points'}
    main_config = MagicMock()
    out_result, out_coords = sm.load_point_poly_data(shap_config, main_config)
    assert out_result['P1'].tolist() == [[1]]
    assert out_coords['P2'].tolist() == [[110.0, 210.0]]


@patch('uncoverml.shapley.features.gather_features')
@patch('uncoverml.shapley.features.transform_features')
@patch('uncoverml.shapley.gen_poly_from_point')
def test_gen_poly_data(mock_genpoly, mock_transform, mock_gather):
    from uncoverml import shapley as sm
    shap_config = MagicMock()
    shap_config.shapefile = {'size': 5}
    feature_set = MagicMock()
    feature_set.transform_set = 'transform_set_A'
    main_config = MagicMock()
    main_config.feature_sets = [feature_set]
    main_config.final_transform = 'final_A'
    image_chunks = {'/some/path.tif': np.array([[1]])}
    coords = np.array([[123, 456]])
    mock_genpoly.return_value = (image_chunks, coords)
    transformed = {True: np.array([[10]])}
    mock_transform.return_value = (transformed, True)
    mock_gather.return_value = np.array([[10]])
    single_row_df = gpd.GeoDataFrame({'Name': ['P1'], 'geometry': [MagicMock()]})
    x, c = sm.gen_poly_data(single_row_df, shap_config, main_config)
    assert x.tolist() == [[10]]
    assert c.tolist() == [[123, 456]]

@patch('uncoverml.shapley.intersect_point_neighbourhood')
@patch('uncoverml.shapley.mpiops.count')
@patch('uncoverml.shapley.mpiops.comm')
@patch('uncoverml.shapley.missing_percentage')
@patch('uncoverml.shapley.log.info')
def test_gen_poly_from_point(mock_log, mock_missing_pct, mock_comm, mock_count, mock_intersect):
    class DummySet:
        def __init__(self, files):
            self.files = files
    class DummyMainConfig:
        feature_sets = [DummySet(files=['/tmp/feat1.tif', '/tmp/feat2.tif'])]
        final_transform = None
    class DummyShapConfig:
        feature_path = '/tmp/'
    gdf = gpd.GeoDataFrame({'geometry': [Point(0, 0)]}, geometry='geometry')
    dummy_data = np.array([1, 2, 3, 4])
    dummy_coords = np.array([[100, 200]])
    mock_intersect.return_value = (dummy_data, dummy_coords)
    mock_count.return_value = np.array([4])
    mock_missing_pct.return_value = 0.0
    mock_comm.allreduce.return_value = 0.0
    results, coords = sm.gen_poly_from_point(gdf, DummyMainConfig(), 3, DummyShapConfig())
    assert isinstance(results, list)
    assert isinstance(results[0], OrderedDict)
    assert list(coords.keys()) == ['feat1', 'feat2']
    for value in results[0].values():
        assert value.shape == (4, 1, 1, 1)

@patch('uncoverml.shapley.rasterio.windows.transform')
@patch('uncoverml.shapley.rasterio.open')
def test_intersect_point_neighbourhood(mock_rio_open, mock_win_transform):
    gdf = gpd.GeoDataFrame({'geometry': [Point(100, 200)]}, geometry='geometry')
    dummy_src = MagicMock()
    dummy_src.index.return_value = (10, 15)
    dummy_image = np.array([[[1.0, 2.0], [3.0, np.nan]]])
    dummy_src.read.return_value = dummy_image
    dummy_src.transform = MagicMock()
    mock_rio_open.return_value.__enter__.return_value = dummy_src
    dummy_transform = MagicMock()
    mock_win_transform.return_value = dummy_transform
    with patch('uncoverml.shapley.rasterio.transform.xy', side_effect=lambda t, r, c, offset: (c + 0.5, r + 0.5)):
        out_image, lon_lat = sm.intersect_point_neighbourhood(gdf, 2, '/tmp/fake.tif')

    assert out_image.shape == (1, 2, 2)
    assert lon_lat[0].shape == (3,)
    assert lon_lat[1].shape == (3,)

def test_to_scientific_notation_basic():
    assert sm.to_scientific_notation(1) == "0.10000E+01"
    val = 12345
    s = sm.to_scientific_notation(val)
    assert "E" in s
    mantissa, exponent = s.split("E")
    assert "." in mantissa and len(mantissa.split(".")[-1]) == 5
    assert (exponent.startswith("+") or exponent.startswith("-")) and len(exponent) == 3


@pytest.mark.parametrize("num, expected_pairs", [
    (1, [(1, 1)]),
    (4, [(1, 4), (2, 2)]),
    (16, [(1, 16), (2, 8), (4, 4)]),
    (18, [(1, 18), (2, 9), (3, 6)]),
])
def test_gen_factors(num, expected_pairs):
    result = sm.gen_factors(num)
    result_int = [(i, int(j)) for (i, j) in result]
    assert set(result_int) == set(expected_pairs)


@pytest.mark.parametrize("n_subplots, expected_grid", [
    (1, (1, 1)),
    (2, (2, 2)),
    (4, (2, 2)),
    (6, (3, 3)),
    (7, (3, 3)),
    (8, (3, 3)),
    (9, (3, 3)),
])
def test_select_subplot_grid_dims(n_subplots, expected_grid):
    r, c = sm.select_subplot_grid_dims(n_subplots)
    assert (r, c) == expected_grid


def test_common_x_text_map_contents():
    assert set(sm.common_x_text_map.keys()) == {"summary", "bar"}
    for txt in sm.common_x_text_map.values():
        assert isinstance(txt, str)


def test_agg_maps_contain_expected_keys():
    assert set(sm.agg_sub_map.keys()) == {"summary", "bar"}
    assert set(sm.agg_sep_map.keys()) == {"decision", "shap_corr"}


def test_explainer_and_masker_maps_have_expected_structure():
    for key, val in sm.explainer_map.items():
        assert isinstance(key, str)
        assert "function" in val and callable(val["function"])
        assert "requirements" in val and isinstance(val["requirements"], list)
        assert "allowed" in val and isinstance(val["allowed"], list)

    for key, val in sm.masker_map.items():
        assert isinstance(key, str)
        assert callable(val)


def test_select_masker_data_type():
    dummy = np.arange(6).reshape((3, 2))
    returned = sm.select_masker("data", dummy)
    assert np.array_equal(returned, dummy)
    assert sm.select_masker("nonexistent", dummy) is None


@pytest.mark.skipif(
    not all(hasattr(cls, "__call__") for cls in (sm.masker_map.get("independent", None), sm.masker_map.get("partition", None))),
    reason="shap.maskers.Independent or Partition not available"
)
def test_select_masker_independent_and_partition():
    small = np.random.rand(10, 4)

    indep = sm.select_masker("independent", small)
    from shap.maskers import Independent as IndepClass
    assert isinstance(indep, IndepClass)

    part = sm.select_masker("partition", small)
    from shap.maskers import Partition as PartClass
    assert isinstance(part, PartClass)


def test_save_plot_creates_file(tmp_path):
    class DummyConfig:
        def __init__(self, p):
            self.output_path = str(p)

    outdir = tmp_path / "plots"
    outdir.mkdir()
    cfg = DummyConfig(outdir)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])

    sm.save_plot(fig, "testplot", cfg)
    plt.close(fig)

    saved = outdir / "testplot.png"
    assert saved.exists() and saved.is_file()


def test_plotconfig_happy_path():
    d = {
        "plot_name": "example",
        "type": "summary",
        "plot_title": "A Title",
        "output_idx": 5,
        "plot_features": ["f1", "f2"],
        "xlim": (0, 1),
        "ylim": (0, 1),
    }
    pc = sm.PlotConfig(d)
    assert pc.plot_name == "example"
    assert pc.type == "summary"
    assert pc.plot_title == "A Title"
    assert pc.output_idx == 5
    assert pc.plot_features == ["f1", "f2"]
    assert pc.xlim == (0, 1)
    assert pc.ylim == (0, 1)


def test_plotconfig_missing_required_keys(caplog):
    caplog.set_level(logging.ERROR)

    d1 = {"type": "summary"}
    _ = sm.PlotConfig(d1)
    assert "Plot name is need to uniquely identify plots" in caplog.text

    caplog.clear()

    d2 = {"plot_name": "onlyname"}
    _ = sm.PlotConfig(d2)
    assert "Need to specify a plot type" in caplog.text


def test_select_masker_independent():
    dummy_data = np.random.rand(10, 5)
    masker = sm.select_masker('independent', dummy_data)
    from shap.maskers import Independent
    assert isinstance(masker, Independent)


def test_prepare_check_masker_list_type():
    dummy_data = np.random.rand(10, 3)
    shap_config = MagicMock()
    shap_config.masker = {
        'type': 'list',
        'mask_list': ['data', 'data', 'data'],
        'start_row': 0,
        'end_row': 10
    }
    shap_config.explainer = 'explainer'
    result = sm.prepare_check_masker(shap_config, dummy_data)
    assert isinstance(result, list)
    assert all(np.array_equal(x, dummy_data) for x in result)


def test_gather_explainer_req_with_masker():
    dummy_data = np.random.rand(10, 5)
    shap_config = MagicMock()
    shap_config.explainer = 'explainer'
    shap_config.masker = {'type': 'data'}
    result = sm.gather_explainer_req(shap_config, dummy_data)
    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    assert result[1] == 0
