import os
import pytest
import click
from click.testing import CliRunner
from uncoverml.scripts import resample_cli as script


class DummyGDF:
    def __init__(self):
        self.saved_path = None
    def to_file(self, output_path):
        self.saved_path = output_path
        open(output_path, "w").close()


class DummyConfig:
    def __init__(self, value_args=None, spatial_args=None, target_file="input.shp", resampled_output="output.shp", target_property="prop"):
        self.value_resampling_args = value_args
        self.spatial_resampling_args = spatial_args
        self.target_file = target_file
        self.resampled_output = resampled_output
        self.target_property = target_property
        self.resample = False


@pytest.fixture(autouse=True)
def patch_config_and_resampling(monkeypatch, tmp_path):
    monkeypatch.setattr(script.ls.config, "Config", lambda path: DummyConfig())
    def dummy_resample_by_magnitude(input_shp, target_field, **kwargs):
        return DummyGDF()
    def dummy_resample_spatially(input_shp, target_field, **kwargs):
        return DummyGDF()
    monkeypatch.setattr(script.ls.resampling, "resample_by_magnitude", dummy_resample_by_magnitude)
    monkeypatch.setattr(script.ls.resampling, "resample_spatially", dummy_resample_spatially)
    monkeypatch.chdir(tmp_path)
    yield


def test_cli_value_resampling_only(monkeypatch, tmp_path):
    value_args = {'threshold': 0.5, 'method': 'median'}
    cfg = DummyConfig(
        value_args=value_args,
        spatial_args=None,
        target_file="in.shp",
        resampled_output="out_value.shp",
        target_property="PROP_VAL"
    )
    monkeypatch.setattr(script.ls.config, "Config", lambda path: cfg)
    called = {'args': None, 'kwargs': None}
    def fake_resample_by_magnitude(input_shp, target_field, **kwargs):
        called['args'] = (input_shp, target_field)
        called['kwargs'] = kwargs
        return DummyGDF()
    monkeypatch.setattr(script.ls.resampling, "resample_by_magnitude", fake_resample_by_magnitude)
    def fake_spatial(*args, **kwargs):
        raise AssertionError("resample_spatially should not be called")
    monkeypatch.setattr(script.ls.resampling, "resample_spatially", fake_spatial)
    dummy_config_file = tmp_path / "pipeline.yaml"
    dummy_config_file.write_text("dummy")
    runner = CliRunner()
    result = runner.invoke(
        script.cli,
        [str(dummy_config_file), "--verbosity", "INFO"],
        catch_exceptions=False
    )
    assert result.exit_code == 0
    assert called['args'] == ("in.shp", "PROP_VAL")
    assert called['kwargs'] == value_args
    assert os.path.exists("out_value.shp")


def test_cli_spatial_resampling_only(monkeypatch, tmp_path):
    spatial_args = {'grid_size': 100, 'interpolation': 'linear'}
    cfg = DummyConfig(
        value_args=None,
        spatial_args=spatial_args,
        target_file="sp_in.shp",
        resampled_output="out_spatial.shp",
        target_property="TARGET_FIELD"
    )
    monkeypatch.setattr(script.ls.config, "Config", lambda path: cfg)
    called = {'args': None, 'kwargs': None}
    def fake_spatial(input_shp, target_field, **kwargs):
        called['args'] = (input_shp, target_field)
        called['kwargs'] = kwargs
        return DummyGDF()
    monkeypatch.setattr(script.ls.resampling, "resample_spatially", fake_spatial)
    def fake_value(*args, **kwargs):
        raise AssertionError("resample_by_magnitude should not be called")
    monkeypatch.setattr(script.ls.resampling, "resample_by_magnitude", fake_value)
    dummy_config_file = tmp_path / "pipeline.yaml"
    dummy_config_file.write_text("dummy")
    runner = CliRunner()
    result = runner.invoke(
        script.cli,
        [str(dummy_config_file), "--verbosity", "INFO"],
        catch_exceptions=False
    )
    assert result.exit_code == 0
    assert called['args'] == ("sp_in.shp", "TARGET_FIELD")
    assert called['kwargs'] == spatial_args
    assert os.path.exists("out_spatial.shp")


def test_cli_both_resampling(monkeypatch, tmp_path):
    value_args = {'threshold': 1.0}
    spatial_args = {'grid_size': 50}
    cfg = DummyConfig(
        value_args=value_args,
        spatial_args=spatial_args,
        target_file="both_in.shp",
        resampled_output="out_both.shp",
        target_property="FIELD"
    )
    monkeypatch.setattr(script.ls.config, "Config", lambda path: cfg)
    call_order = []
    def fake_value(input_shp, target_field, **kwargs):
        call_order.append(("value", input_shp, target_field, kwargs))
        return DummyGDF()
    def fake_spatial(input_shp, target_field, **kwargs):
        call_order.append(("spatial", input_shp, target_field, kwargs))
        return DummyGDF()
    monkeypatch.setattr(script.ls.resampling, "resample_by_magnitude", fake_value)
    monkeypatch.setattr(script.ls.resampling, "resample_spatially", fake_spatial)
    dummy_config_file = tmp_path / "pipeline.yaml"
    dummy_config_file.write_text("dummy")
    runner = CliRunner()
    result = runner.invoke(
        script.cli,
        [str(dummy_config_file), "--verbosity", "INFO"],
        catch_exceptions=False
    )
    assert result.exit_code == 0
    assert call_order[0][0] == "value"
    assert call_order[0][1:] == ("both_in.shp", "FIELD", value_args)
    assert call_order[1][0] == "spatial"
    assert call_order[1][1:] == ("both_in.shp", "FIELD", spatial_args)
    assert os.path.exists("out_both.shp")


def test_cli_no_transforms(monkeypatch, tmp_path):
    cfg = DummyConfig(
        value_args=None,
        spatial_args=None,
        target_file="none_in.shp",
        resampled_output="none_out.shp",
        target_property="X"
    )
    monkeypatch.setattr(script.ls.config, "Config", lambda path: cfg)
    dummy_config_file = tmp_path / "pipeline.yaml"
    dummy_config_file.write_text("dummy")

    runner = CliRunner()
    result = runner.invoke(
        script.cli,
        [str(dummy_config_file), "--verbosity", "INFO"],
        catch_exceptions=True
    )

    assert result.exit_code != 0
    exc_msg = str(result.exception)
    assert "NoneType" in exc_msg and "to_file" in exc_msg
