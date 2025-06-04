# tests/test_optimisation_module.py
import json
import os
import pandas as pd
import numpy as np
import pytest

import uncoverml.hyopt as om


class DummyConfig:
    def __init__(self, tmpdir):
        self.hp_params_space = {}
        self.algorithm = "dummy_algo"
        self.algorithm_args = {}
        self.hyperopt_params = {
            "random_state": 0,
            "scoring": "r2",
        }
        self.output_dir = str(tmpdir / "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.optimised_model_params = str(tmpdir / "best_params.json")
        self.optimisation_output_hpopt = str(tmpdir / "hpopt_results.csv")
        self.optimised_model = False
        self.bayes_or_anneal = "bayes"
        self.algo = lambda: None


class DummyTargets:
    def __init__(self, n_samples=10):
        self.observations = np.arange(n_samples).reshape(-1, 1)
        self.positions = np.vstack([np.linspace(0, 1, n_samples),
                                    np.linspace(0, 1, n_samples)]).T
        self.groups = np.zeros(n_samples, dtype=int)


@pytest.fixture(autouse=True)
def patch_outside_dependencies(monkeypatch, tmp_path):
    class DummyEstimator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.random_state = kwargs.get("random_state", None)
        def fit(self, X, y, sample_weight=None):
            self.fitted_ = True
        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setitem(om.modelmaps, "dummy_algo", DummyEstimator)

    def fake_setup_validation_data(X, targets_all, cv_folds, random_state):
        X_dummy = np.random.rand(10, 3)
        y_dummy = targets_all.observations.flatten()
        lon_lat_dummy = targets_all.positions
        groups_dummy = targets_all.groups
        w_dummy = np.ones_like(y_dummy)
        from sklearn.model_selection import KFold
        cv_dummy = KFold(n_splits=2, shuffle=True, random_state=random_state)
        return X_dummy, y_dummy, lon_lat_dummy, groups_dummy, w_dummy, cv_dummy

    monkeypatch.setattr(om, "setup_validation_data", fake_setup_validation_data)

    def fake_cross_validate(model, X, y, fit_params, groups, cv, scoring, n_jobs):
        n_splits = cv.get_n_splits(X, y)
        return {"test_score": np.array([0.8] * n_splits)}

    monkeypatch.setattr(om, "cross_validate", fake_cross_validate)

    def fake_fmin(fn, space, **kwargs):
        return {}

    monkeypatch.setattr(om, "fmin", fake_fmin)

    def fake_space_eval(space, best):
        return best

    monkeypatch.setattr(om, "space_eval", fake_space_eval)

    called = {"exported": False}

    def fake_export_model(model, conf, flag):
        called["exported"] = True

    monkeypatch.setattr(om.geoio, "export_model", fake_export_model)
    monkeypatch.setattr(om, "write_progress_to_file", lambda *args, **kwargs: None)

    return called


def test_save_optimal_creates_json_and_csv(tmp_path):
    trials = [
        {"misc": {"vals": {"a": [1], "b": [10.0]}}, "result": {"loss": 0.5}},
        {"misc": {"vals": {"a": [2], "b": [20.0]}}, "result": {"loss": 0.3}}
    ]

    class FakeTrials:
        def __init__(self, trials_list):
            self.trials = trials_list

    cfg = DummyConfig(tmp_path)
    best = {"a": 2, "b": 20.0}

    def dummy_obj(x):
        return 1.0 if x.get("a", 0) == 2 else 2.0

    om.save_optimal(best=best,
                    random_state=0,
                    trials=FakeTrials(trials),
                    objective=dummy_obj,
                    conf=cfg)

    with open(cfg.optimised_model_params, "r") as f:
        loaded = json.load(f)

    assert loaded["a"] == 2
    assert loaded["b"] == 20.0

    assert os.path.exists(cfg.optimisation_output_hpopt)
    df = pd.read_csv(cfg.optimisation_output_hpopt)

    assert "loss" in df.columns
    assert "a" in df.columns
    assert "b" in df.columns

    assert len(df) == 2
    assert pytest.approx(df["loss"].iloc[0]) == 0.3
    assert pytest.approx(df["loss"].iloc[1]) == 0.5


def test_optimise_model_integration(tmp_path, patch_outside_dependencies):
    X_dummy = np.random.rand(10, 5)
    targets = DummyTargets(n_samples=10)
    config = DummyConfig(tmp_path)

    om.optimise_model(X_dummy, targets, config)

    assert os.path.exists(config.optimised_model_params)
    assert os.path.exists(config.optimisation_output_hpopt)
    assert patch_outside_dependencies["exported"] is True
    assert config.optimised_model is True
