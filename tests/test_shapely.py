# tests/test_shap_module.py

import logging
import pytest
import numpy as np
import matplotlib.pyplot as plt
import uncoverml.shapley as sm


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
