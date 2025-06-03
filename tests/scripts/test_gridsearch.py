import os
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from click.testing import CliRunner

from uncoverml.scripts import gridsearch
from uncoverml.config import ConfigException


class DummyConfig:
    def __init__(self):
        self.optimisation = {}
        self.optimisation_output = ""
        self.n_jobs = 1


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    yield


def test_setup_pipeline_invalid_algorithm():
    cfg = DummyConfig()
    cfg.optimisation['algorithm'] = "nonexistent_algo"

    with pytest.raises(ConfigException):
        gridsearch.setup_pipeline(cfg)


def test_setup_pipeline_minimal_algorithm_only():
    cfg = DummyConfig()
    cfg.optimisation['algorithm'] = "transformedgp"
    cfg.n_jobs = 2

    estimator = gridsearch.setup_pipeline(cfg)
    assert isinstance(estimator, GridSearchCV)
    assert estimator.param_grid == {}
    assert isinstance(estimator.estimator, Pipeline)
    assert estimator.estimator.steps == []


def test_setup_pipeline_with_simple_hyperparameter():
    cfg = DummyConfig()
    cfg.optimisation['algorithm'] = "transformedsvr"
    cfg.n_jobs = 1

    cfg.optimisation['hyperparameters'] = {'C': [0.1, 1.0]}

    estimator = gridsearch.setup_pipeline(cfg)
    assert isinstance(estimator, GridSearchCV)

    expected_key = "transformedsvr__C"
    assert expected_key in estimator.param_grid
    assert estimator.param_grid[expected_key] == [0.1, 1.0]

    steps = estimator.estimator.steps
    assert len(steps) == 1
    name, transformer = steps[0]
    assert name == "transformedsvr"
    from uncoverml.optimise.models import TransformedSVR
    assert isinstance(transformer, TransformedSVR)


def test_setup_pipeline_with_pca_and_hyperparameters(monkeypatch):
    from sklearn.decomposition import PCA

    cfg = DummyConfig()
    cfg.optimisation['algorithm'] = "transformedsvr"
    cfg.n_jobs = -1

    cfg.optimisation['featuretransforms'] = {'pca': {'n_components': [2, 3]}}
    cfg.optimisation['hyperparameters'] = {'C': [10, 100]}

    estimator = gridsearch.setup_pipeline(cfg)

    pipeline_steps = estimator.estimator.steps
    assert len(pipeline_steps) == 2
    assert pipeline_steps[0][0] == "pca"
    assert isinstance(pipeline_steps[0][1], PCA)
    assert pipeline_steps[1][0] == "transformedsvr"

    keys = set(estimator.param_grid.keys())
    assert "pca__n_components" in keys
    assert "transformedsvr__C" in keys

    assert estimator.param_grid["pca__n_components"] == [2, 3]
    assert estimator.param_grid["transformedsvr__C"] == [10, 100]


class FakeEstimator:
    def __init__(self):
        self.cv_results_ = None
        self.fit_called = False

    def fit(self, X, y):
        self.fit_called = True
        self.cv_results_ = {
            "param_a": [1, 2],
            "mean_test_score": [0.8, 0.9],
            "rank_test_score": [2, 1],
        }


def test_cli_writes_csv(monkeypatch, tmp_path):
    cfg = DummyConfig()
    cfg.optimisation['algorithm'] = "myalgo"
    cfg.optimisation_output = "out.csv"
    cfg.n_jobs = 1

    monkeypatch.setattr(gridsearch.ls.config, "Config", lambda path: cfg)

    fake_est = FakeEstimator()
    def fake_setup_pipeline(config_arg):
        assert config_arg is cfg
        return fake_est
    monkeypatch.setattr(gridsearch, "setup_pipeline", fake_setup_pipeline)

    dummy_targets = type("T", (), {"observations": np.array([1, 2, 3])})
    dummy_X = np.array([[0], [1], [2]])
    def fake_load_data(config_arg, partitions_arg):
        assert config_arg is cfg
        assert partitions_arg == 1
        return dummy_targets, dummy_X
    monkeypatch.setattr(gridsearch, "_load_data", fake_load_data)

    runner = CliRunner()
    dummy_pipeline_file = tmp_path / "pipeline.yaml"
    dummy_pipeline_file.write_text("dummy: content")

    result = runner.invoke(
        gridsearch.cli,
        [
            str(dummy_pipeline_file),
            "--partitions", "1",
            "--njobs", "1",
            "--verbosity", "INFO",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert fake_est.fit_called

    expected_csv = tmp_path / "myalgo_out.csv"
    assert expected_csv.exists()

    df = pd.read_csv(expected_csv, index_col=0)
    assert list(df["param_a"]) == [2, 1]
    assert list(df["rank_test_score"]) == [1, 2]


def test_cli_config_exception(monkeypatch):
    def raise_config(path):
        raise ConfigException("bad config")

    monkeypatch.setattr(gridsearch.ls.config, "Config", raise_config)
    runner = CliRunner()
    result = runner.invoke(
        gridsearch.cli,
        [
            "dummy_path.yaml",
            "--partitions", "1",
            "--njobs", "1",
            "--verbosity", "INFO",
        ],
        catch_exceptions=True
    )
    assert result.exit_code != 0
