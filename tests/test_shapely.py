# tests/test_shap_module.py

import logging
import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
import geopandas as gpd
import os
from collections import OrderedDict
import uncoverml.shapley as sm
from unittest.mock import patch, MagicMock, mock_open
from shapely.geometry import Point, Polygon
from types import SimpleNamespace
from shap.maskers import Independent, Partition


@patch("uncoverml.shapley.rasterio.transform.xy")
@patch("uncoverml.shapley.mask")
@patch("uncoverml.shapley.rasterio.open")
def test_intersect_shp_poly(mock_rio_open, mock_mask, mock_xy):
    poly = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, geometry="geometry")
    dummy_src = MagicMock()
    dummy_src.nodata = -9999
    mock_rio_open.return_value.__enter__.return_value = dummy_src
    out_image = np.array([[[[1.0, 2.0], [np.nan, np.nan]]]])
    dummy_transform = object()
    mock_mask.return_value = (out_image, dummy_transform)
    mock_xy.side_effect = lambda t, r, c, offset=None: (c + 100, r + 200)
    img_path = "/test/feature.tif"
    out_img, lonlat, shp = sm.intersect_shp(gdf, img_path, type="poly")
    assert shp == (1, 2, 2)
    assert isinstance(lonlat, tuple) and len(lonlat) == 2
    lons, lats = lonlat
    assert np.array_equal(np.array(lons), np.array([100, 101]))
    assert np.array_equal(np.array(lats), np.array([200, 200]))


@patch("uncoverml.shapley.mask")
@patch("uncoverml.shapley.rasterio.open")
def test_intersect_shp_nonpoly(mock_rio_open, mock_mask):
    pt = Point(1, 1)
    gdf = gpd.GeoDataFrame({"geometry": [pt]}, geometry="geometry")
    dummy_src = MagicMock()
    mock_rio_open.return_value.__enter__.return_value = dummy_src
    out_image = np.array([[[[5.0, np.nan], [np.nan, 7.0]]]])
    dummy_transform = object()
    mock_mask.return_value = (out_image, dummy_transform)
    out_img, lonlat, shp = sm.intersect_shp(gdf, "/fake/feature.tif", type="points")
    assert shp == (1, 2, 2)
    assert lonlat is None


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
    assert isinstance(indep, Independent)

    part = sm.select_masker("partition", small)
    assert isinstance(part, Partition)


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


@patch("uncoverml.shapley.PlotConfig", autospec=True)
@patch("uncoverml.shapley.yaml.safe_load")
@patch("builtins.open", new_callable=mock_open, read_data="dummy")
def test_shapconfig(mock_file, mock_yaml, mock_plotconfig):
    class DummySet:
        def __init__(self, files):
            self.files = files
    yaml_dict = {
        "explainer": "kernel",
        "shapefile": {"type": "poly", "dir": "/some/shape"},
        "explainer_kwargs": {"l1_reg": "num_features(5)"},
        "calc_start_row": 10,
        "calc_end_row": 20,
        "masker": {"type": "data"},
        "output_path": "/custom/out",
        "plots": [{"plot_name": "p1", "type": "summary"}, {"plot_name": "p2", "type": "bar"}],
        "output_names": ["Prediction"],
        "load_file": "/some/shap_values.npz",
        "save": {"name": "save_tag"},
        "feature_names": ["f_short1", "f_short2", "f_short3"],
    }
    mock_yaml.return_value = yaml_dict
    main_config = MagicMock()
    main_config.output_dir = "/main/out"
    main_config.feature_sets = [
        DummySet(files=["/features/f1.tif", "/features/f2.tif"]),
        DummySet(files=["/features/f3.tif"]),
    ]
    cfg_path = "/path/to/mycfg.yaml"
    sc = sm.ShapConfig(cfg_path, main_config)
    assert sc.name == "mycfg"
    assert sc.explainer == "kernel"
    assert sc.shapefile == {"type": "poly", "dir": "/some/shape"}
    assert sc.explainer_kwargs == {"l1_reg": "num_features(5)"}
    assert sc.calc_start_row == 10
    assert sc.calc_end_row == 20
    assert sc.masker == {"type": "data"}
    assert sc.output_path == "/custom/out"
    assert sc.output_names == ["Prediction"]
    assert sc.load_file == "/some/shap_values.npz"
    assert sc.do_save is True and sc.save_name == "save_tag"
    assert mock_plotconfig.call_count == 2
    assert isinstance(sc.plot_config_list, list) and len(sc.plot_config_list) == 2
    assert sc.file_names == ["f1.tif", "f2.tif", "f3.tif"]
    assert sc.feature_names == ["f_short1", "f_short2", "f_short3"]


@patch("uncoverml.shapley.PlotConfig", autospec=True)
@patch("uncoverml.shapley.yaml.safe_load")
@patch("builtins.open", new_callable=mock_open, read_data="dummy")
def test_shapconfig_fallbacks_logs_and_paths(mock_file, mock_yaml, mock_plotconfig, caplog):
    class DummySet:
        def __init__(self, files):
            self.files = files
    yaml_dict = {"feature_names": ["fx", "fy"]}
    mock_yaml.return_value = yaml_dict
    main_config = MagicMock()
    main_config.output_dir = "/project/out"
    main_config.feature_sets = [
        DummySet(files=["/a/b/featA.tif"]),
        DummySet(files=["/a/b/featB.tif"]),
    ]
    caplog.set_level(logging.WARNING, logger="uncoverml.shapley")
    sc = sm.ShapConfig("/tmp/conf.yml", main_config)
    assert sc.explainer is None
    assert sc.shapefile is None
    assert sc.output_path == os.path.join(main_config.output_dir, "shap")
    assert sc.do_save is False
    assert not hasattr(sc, "save_name") or sc.save_name is None
    assert sc.file_names == ["featA.tif", "featB.tif"]
    assert sc.feature_names == ["fx", "fy"]
    assert sc.plot_config_list is None
    assert "No plots will be created" in caplog.text
    assert "No explainer provided, cannot calculate Shapley values" in caplog.text
    assert "No shapefile provided, calculation will fail" in caplog.text


def stub_predict(x, model=None, **kwargs):
    return np.sum(np.asarray(x), axis=1, keepdims=True)


@patch("uncoverml.shapley.gather_explainer_req", autospec=True)
@patch("uncoverml.shapley.explainer_map", new={})
def test_calc_shap_vals_invalid_explainer(mock_gather):
    x = np.arange(6).reshape(3, 2)
    shap_config = MagicMock()
    shap_config.explainer = 'does_not_exist'
    shap_config.explainer_kwargs = None
    shap_config.calc_start_row = None
    shap_config.calc_end_row = None
    with patch.object(sm, "log") as mock_log:
        out = sm.calc_shap_vals(model=None, shap_config=shap_config, x_data=x)
        assert out is None
        assert any("Invalid or no explainer specified" in (args[0] if args else "")
                   for (name, args, _) in mock_log.method_calls)


@patch("uncoverml.shapley.predict.predict", side_effect=stub_predict, autospec=True)
def test_calc_shap_vals_with_kwargs_and_warn_on_unfulfilled(mock_predict, caplog):
    captured = {}
    def factory(shap_predict, *reqs, **kwargs):
        captured["reqs"] = reqs
        captured["kwargs"] = kwargs
        def explainer_obj(data):
            return {"preds": shap_predict(data), "kw": kwargs}
        return explainer_obj
    with patch("uncoverml.shapley.explainer_map",
               new={"dummy": {"function": factory, "requirements": [], "allowed": []}}), \
         patch("uncoverml.shapley.gather_explainer_req",
               return_value=(["need_this"], 2), autospec=True):
        caplog.set_level(logging.WARNING)
        x = np.arange(8).reshape(4, 2)
        shap_config = MagicMock()
        shap_config.explainer = 'dummy'
        shap_config.explainer_kwargs = {"alpha": 0.5, "n_samples": 20}
        shap_config.calc_start_row = None
        shap_config.calc_end_row = None
        out = sm.calc_shap_vals(model=None, shap_config=shap_config, x_data=x)
        assert captured["kwargs"] == {"alpha": 0.5, "n_samples": 20}
        assert captured["reqs"] == ("need_this",)
        assert any("Some explainer requirements not fulfilled" in rec.message for rec in caplog.records)
        assert out["preds"].shape == (4, 1)
        assert mock_predict.called


@patch("uncoverml.shapley.save_plot", autospec=True)
@patch("uncoverml.shapley.common_x_text_map", new={"bar": "common X text", "summary": "common X text"})
@patch("uncoverml.shapley.agg_sub_map")
@patch("uncoverml.shapley.plt.subplots")
def test_aggregate_subplot_calls_mapper_and_saves(mock_subplots, mock_agg_sub_map, mock_save_plot):
    fig = MagicMock()
    ax0 = MagicMock()
    ax1 = MagicMock()
    def _fake_subplots(nrows, ncols, dpi=100):
        axs = [ax0, ax1][:ncols]
        return fig, axs
    mock_subplots.side_effect = _fake_subplots
    calls = []
    def _handler(plot_data, ax, idx, **kwargs):
        calls.append((idx, kwargs.get("output_name")))
    mock_agg_sub_map.__getitem__.side_effect = lambda k: _handler
    plot_vals = np.zeros((5, 3, 2))
    shap_cfg = MagicMock()
    sm.aggregate_subplot(plot_vals, plot_type="bar", shap_config=shap_cfg, output_names=["out0", "out1"])
    assert calls == [(0, "out0"), (1, "out1")]
    mock_save_plot.assert_called_once()
    args, _ = mock_save_plot.call_args
    assert args[1] == "bar" and args[2] is shap_cfg
    assert fig.text.called


@patch("uncoverml.shapley.save_plot", autospec=True)
@patch("uncoverml.shapley.agg_sep_map")
@patch("uncoverml.shapley.plt.subplots")
def test_aggregate_separate_calls_mapper_each_output_and_saves_each(mock_subplots, mock_agg_sep_map, mock_save_plot):
    fig = MagicMock()
    ax = MagicMock()
    mock_subplots.return_value = (fig, ax)
    calls = []
    def _handler(plot_data, ax_in, idx, **kwargs):
        calls.append((idx, kwargs.get("output_name")))
    mock_agg_sep_map.__getitem__.side_effect = lambda k: _handler
    plot_vals = np.zeros((4, 2, 3))
    shap_cfg = SimpleNamespace()
    sm.aggregate_separate(plot_vals, plot_type="shap_corr", shap_config=shap_cfg,
                          output_names=["o0", "o1", "o2"])
    assert calls == [(0, "o0"), (1, "o1"), (2, "o2")]
    assert mock_save_plot.call_count == 3
    for call in mock_save_plot.call_args_list:
        assert call.args[1] == "shap_corr"
        assert call.args[2] is shap_cfg


@patch("uncoverml.shapley.shap.summary_plot", autospec=True)
@patch("uncoverml.shapley.plt.gcf")
def test_summary_plot(mock_gcf, mock_summary_plot):
    fake_fig = MagicMock()
    fake_colorbar_ax = MagicMock()
    fake_fig.axes = [fake_colorbar_ax]
    mock_gcf.return_value = fake_fig
    target_ax = MagicMock()
    target_ax.axes.get_xaxis.return_value.get_label.return_value = MagicMock()
    class PlotData:
        def __init__(self):
            self.values = np.random.rand(10, 3)
            self.data = np.random.rand(10, 3)
            self.feature_names = ["f1", "f2", "f3"]
        def shape(self):
            return self.values.shape
        shape = (10, 3)
    pd_obj = PlotData()
    sm.summary_plot(pd_obj, target_ax, plot_idx=1, output_name="PRED")
    mock_summary_plot.assert_called_once()
    args, kwargs = mock_summary_plot.call_args
    assert kwargs["features"] is pd_obj.data
    assert kwargs["feature_names"] == pd_obj.feature_names
    assert kwargs["show"] is False
    target_ax.title.set_text.assert_called_once()
    assert "PRED" in target_ax.title.set_text.call_args.args[0]
    target_ax.axes.get_xaxis.return_value.get_label.return_value.set_visible.assert_called_once_with(False)
    target_ax.axes.yaxis.set_visible.assert_called_once_with(False)


@patch("uncoverml.shapley.shap.plots.bar", autospec=True)
def test_bar_plot_calls_shap_bar_and_sets_title(mock_shap_bar):
    target_ax = MagicMock()
    class PlotData:
        shape = (12, 4)
    pd_obj = PlotData()
    sm.bar_plot(pd_obj, target_ax, output_name="OutX")
    mock_shap_bar.assert_called_once()
    target_ax.title.set_text.assert_called_once()
    assert "OutX" in target_ax.title.set_text.call_args.args[0]
    target_ax.axes.get_xaxis.return_value.get_label.return_value.set_visible.assert_called_once_with(False)
    target_ax.tick_params.assert_called()


@patch("uncoverml.shapley.sns.heatmap", autospec=True)
def test_shap_corr_plot_with_and_without_feature_names_kw(mock_heatmap):
    target_ax = MagicMock()
    class PData:
        values = np.random.rand(6, 3)
        feature_names = ["a", "b", "c"]
        shape = (6, 3)
    pd_obj = PData()
    sm.shap_corr_plot(pd_obj, target_ax, output_name="A", feature_names=True)
    target_ax.title.set_text.assert_called()
    assert "A" in target_ax.title.set_text.call_args.args[0]
    sm.shap_corr_plot(pd_obj, target_ax, output_name="B")
    assert mock_heatmap.call_count >= 2
    target_ax.tick_params.assert_called()


@patch("uncoverml.shapley.shap.decision_plot", autospec=True)
def test_decision_plot_calls_shap_and_sets_title(mock_decision_plot):
    target_ax = MagicMock()
    class PData:
        base_values = np.array([0.5])
        values = np.random.rand(5, 3)
        feature_names = ["f1", "f2", "f3"]
    pd_obj = PData()
    sm.decision_plot(pd_obj, target_ax, output_name="D1")
    mock_decision_plot.assert_called_once_with(pd_obj.base_values[0],
                                               pd_obj.values,
                                               feature_names=pd_obj.feature_names)
    target_ax.title.set_text.assert_called_once()
    assert "D1" in target_ax.title.set_text.call_args.args[0]
    target_ax.tick_params.assert_called()


def test_spatial_plot_sets_ticks_and_title():
    class PD:
        values = np.array([1, 2, 3, 4])

    ax = MagicMock()
    lons = np.array([10.0, 20.0])
    lats = np.array([30.0, 40.0])
    out = sm.spatial_plot("featX", ax, PD(), (lons, lats), size=2)
    assert out is not None
    ax.imshow.assert_called_once()
    assert ax.set_xticklabels.call_args[0][0] and len(ax.set_xticklabels.call_args[0][0]) == 4
    assert ax.set_yticklabels.call_args[0][0] and len(ax.set_yticklabels.call_args[0][0]) == 4
    ax.set_title.assert_called_once_with("featX")


@patch("uncoverml.shapley.aggregate_feature_subplots", autospec=True)
@patch("uncoverml.shapley.aggregate_separate", autospec=True)
@patch("uncoverml.shapley.aggregate_subplot", autospec=True)
def test_generate_plots_poly_dispatches_with_output_names(mock_sub, mock_sep, mock_feat, tmp_path):
    class SV:
        def __init__(self):
            self.shape = (10, 4, 2)
            self.feature_names = None
    shap_vals = SV()
    shap_cfg = SimpleNamespace(output_names=["Y0", "Y1"], output_path=str(tmp_path))
    kwargs = {"feature_names": ["a", "b", "c", "d"]}
    sm.generate_plots_poly(shap_vals, shap_cfg, lon_lats={"dummy": None}, **kwargs)
    assert shap_vals.feature_names == kwargs["feature_names"]
    for call in mock_sub.call_args_list:
        assert call.kwargs["output_names"] == ["Y0", "Y1"]
    for call in mock_sep.call_args_list:
        assert call.kwargs["output_names"] == ["Y0", "Y1"]
    for call in mock_feat.call_args_list:
        assert call.kwargs["output_names"] == ["Y0", "Y1"]


@patch("uncoverml.shapley.aggregate_feature_subplots", autospec=True)
@patch("uncoverml.shapley.aggregate_separate", autospec=True)
@patch("uncoverml.shapley.aggregate_subplot", autospec=True)
def test_generate_plots_poly_warns_without_feature_names(mock_sub, mock_sep, mock_feat, caplog, tmp_path):
    caplog.set_level("WARNING")
    class SV:
        def __init__(self):
            self.shape = (5, 2)
            self.feature_names = None
    shap_vals = SV()
    shap_cfg = SimpleNamespace(output_names=None, output_path=str(tmp_path))
    sm.generate_plots_poly(shap_vals, shap_cfg, lon_lats={}, feature_names=None)
    assert "Feature names not provided" in caplog.text
    for call in mock_sub.call_args_list:
        assert call.kwargs["output_names"] is None
    for call in mock_sep.call_args_list:
        assert call.kwargs["output_names"] is None
    for call in mock_feat.call_args_list:
        assert call.kwargs["output_names"] is None


@patch("uncoverml.shapley.spatial_point_poly", autospec=True)
@patch("uncoverml.shapley.point_poly_subplots", autospec=True)
def test_generate_plots_poly_point_sets_feature_names_and_calls_subplots(mock_pps, mock_spatial, caplog):
    caplog.set_level("WARNING")
    class PointContainer:
        def __init__(self, n_points, n_features):
            self._n_points = n_points
            self.feature_names = None
            self._data = np.arange(n_points * n_features).reshape(n_points, n_features)
        def __getitem__(self, idx):
            return SimpleNamespace(data=self._data.copy(),
                                   base_values=np.array([0.5]),
                                   values=np.random.rand(3, 3),
                                   feature_names=None)
    n_points = 2
    shap_vals_dict = {f"P{i+1}": SimpleNamespace(
        values=np.random.rand(6, 3),
        data=np.random.rand(6, 3),
        feature_names=None,
        base_values=np.array([0.5]),
        shape=(6, 3, 1)
    ) for i in range(n_points)}
    shap_vals_point = PointContainer(n_points=n_points, n_features=3)
    shap_cfg = SimpleNamespace(feature_names=["F1", "F2", "F3"], output_path="/tmp/out")
    name_list = ["P1", "P2"]
    sm.generate_plots_poly_point(name_list, shap_vals_dict, shap_vals_point, shap_cfg,
                                 output_names=["Y0"], lon_lats={"P1": None, "P2": None})
    assert shap_vals_point.feature_names == ["F1", "F2", "F3"]
    for v in shap_vals_dict.values():
        assert v.feature_names == ["F1", "F2", "F3"]
    assert mock_pps.call_count == len(name_list)
    assert mock_spatial.call_count == len(name_list)


def test_ax_tidy_point_poly():
    ax = MagicMock()
    ax.get_yticklabels.return_value = [MagicMock(), MagicMock()]
    sm.ax_tidy_point_poly(ax, "A Very Long Plot Title", padding=7)
    assert ax.set_title.called
    ax.tick_params.assert_any_call(axis='both', labelsize=5)
    ax.tick_params.assert_any_call(axis='y', pad=7)
    ax.set_yticklabels.assert_called()
    ax.xaxis.get_label.assert_called()


@patch("uncoverml.shapley.shap.decision_plot", autospec=True)
@patch("uncoverml.shapley.shap.plots.bar", autospec=True)
@patch("uncoverml.shapley.shap.summary_plot", autospec=True)
@patch("uncoverml.shapley.shap.waterfall_plot", autospec=True)
def test_point_poly_subplots(mock_waterfall, mock_summary, mock_bar, mock_decision, tmp_path):
    class PointsVal:
        base_values = np.array([0.5])
        values = np.random.rand(3, 3)
        feature_names = ["f1", "f2", "f3"]
    class PolyVals:
        values = np.random.rand(8, 3)
        data = np.random.rand(8, 3)
        feature_names = ["f1", "f2", "f3"]
        base_values = np.array([0.5])
        shape = (8, 3, 1)
    pv = PointsVal()
    polv = PolyVals()
    shap_cfg = SimpleNamespace(output_path=str(tmp_path))
    sm.point_poly_subplots("SiteA", polv, pv, shap_cfg, output_names=["Y0"])
    assert mock_waterfall.called
    assert mock_summary.called
    assert mock_bar.called
    assert mock_decision.called
    expected = tmp_path / "poly_point_SiteA_Y0.png"
    assert expected.exists()


def make_shap_vals(n_rows=5, n_feats=2, n_out=1):
    class SliceObj:
        def __init__(self, n=4):
            self.values = np.arange(n)
    class SVals:
        def __init__(self):
            self.feature_names = [f"f{i+1}" for i in range(n_feats)]
            self.shape = (n_rows, n_feats, n_out)
        def __getitem__(self, key):
            return SliceObj(n=4)
    return SVals()


@patch("uncoverml.shapley.select_subplot_grid_dims", return_value=(1, 2))
@patch.object(sm, "output_names", ["O0"], create=True)
@patch.object(sm, "point_poly_vals", SimpleNamespace(shape=(3, 2, 1)), create=True)
def test_aggregate_feature_subplots_scatter_saves_png(mock_grid, tmp_path):
    shap_vals = make_shap_vals(n_feats=2, n_out=1)
    shap_cfg = SimpleNamespace(output_path=str(tmp_path), file_names=["f1.tif", "f2.tif"])
    lon_lats = {
        "f1.tif": (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        "f2.tif": (np.array([2.0, 3.0]), np.array([4.0, 5.0])),
    }
    sm.aggregate_feature_subplots(shap_vals, "scatter", shap_cfg, lon_lats, output_names=["O0"], size=2)
    assert (tmp_path / "polygon_scatter_O0.png").exists()


def test_aggregate_feature_subplots_spatial_calls_dependence(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "point_poly_vals", SimpleNamespace(shape=(3, 2, 1)), raising=False)
    monkeypatch.setattr(sm, "output_names", ["O0"], raising=False)
    monkeypatch.setattr(sm, "select_subplot_grid_dims", lambda n: (1, 2), raising=True)
    calls = []
    def dep_plot(feat_name, plot_vals, ax, **kwargs):
        calls.append((feat_name, ax))
    monkeypatch.setattr(sm, "dependence_plot", dep_plot, raising=False)
    shap_vals = make_shap_vals(n_feats=2, n_out=1)
    shap_cfg = SimpleNamespace(output_path=str(tmp_path), file_names=["f1.tif", "f2.tif"])
    lon_lats = {
        "f1.tif": (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        "f2.tif": (np.array([2.0, 3.0]), np.array([4.0, 5.0])),
    }
    sm.aggregate_feature_subplots(shap_vals, "spatial", shap_cfg, lon_lats, output_names=["O0"])
    assert len(calls) == 2
    assert {c[0] for c in calls} == {"f1", "f2"}
    expected = tmp_path / "polygon_spatial_O0.png"
    assert expected.exists(), f"Expected saved figure at {expected}"


def make_point_poly_vals(n_rows=4, n_feats=2, n_out=1):
    class SliceObj:
        def __init__(self, n):
            self.values = np.arange(n)
            self.shape = (n,)
    class PPVals:
        def __init__(self):
            self.feature_names = [f"f{i+1}" for i in range(n_feats)]
            self.shape = (n_rows, n_feats, n_out)
        def __getitem__(self, key):
            return SliceObj(n=4)
    return PPVals()


def test_spatial_point_poly(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "select_subplot_grid_dims", lambda n: (1, 2), raising=True)
    colorbar_calls = []
    def _spy_colorbar(self, *args, **kwargs):
        colorbar_calls.append((args, kwargs))
    monkeypatch.setattr(mfig.Figure, "colorbar", _spy_colorbar, raising=True)
    point_poly_vals = make_point_poly_vals(n_feats=2, n_out=1)
    shap_cfg = SimpleNamespace(output_path=str(tmp_path), file_names=["f1.tif", "f2.tif"])
    lon_lats = {
        "f1.tif": (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        "f2.tif": (np.array([2.0, 3.0]), np.array([4.0, 5.0])),
    }
    sm.spatial_point_poly(
        name="SiteA",
        point_poly_vals=point_poly_vals,
        lon_lats=lon_lats,
        shap_config=shap_cfg,
        output_names=["Y0"],
    )
    assert len(colorbar_calls) == 1
    expected = tmp_path / "spatial_poly_point_SiteA_Y0.png"
    assert expected.exists(), f"Expected figure at {expected}"


def test_spatial_point_poly_default_output_name(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "select_subplot_grid_dims", lambda n: (1, 2), raising=True)
    point_poly_vals = make_point_poly_vals(n_feats=2, n_out=1)
    shap_cfg = SimpleNamespace(output_path=str(tmp_path), file_names=["f1.tif", "f2.tif"])
    lon_lats = {
        "f1.tif": (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        "f2.tif": (np.array([2.0, 3.0]), np.array([4.0, 5.0])),
    }
    sm.spatial_point_poly(
        name="SiteB",
        point_poly_vals=point_poly_vals,
        lon_lats=lon_lats,
        shap_config=shap_cfg,
    )
    expected = tmp_path / "spatial_poly_point_SiteB_0.png"
    assert expected.exists(), f"Expected figure at {expected}"
