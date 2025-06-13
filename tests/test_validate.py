import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold

from uncoverml.validate import (
    _join_dicts,
    split_cfold,
    split_gfold,
    classification_validation_scores,
    regression_validation_scores,
    setup_validation_data,
)

class DummyIdentityTransform:
    def transform(self, x):
        return x


class DummyRegressionModel:
    def __init__(self):
        self.target_transform = DummyIdentityTransform()

    def get_predict_tags(self):
        return ["Prediction"]

class DummyTargets:
    def __init__(self):
        self.observations = np.array([1.0, 2.0, 3.0, 4.0])
        self.weights = np.array([1.0, 1.0, 1.0, 1.0])
        self.positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.groups = np.array([0, 0, 1, 1])
        self.fields = {
            'soil_type': np.array([0, 1, 0, 1])
        }


def test_split_cfold_basic():
    nsamples = 10
    k = 5
    cvinds, cvassigns = split_cfold(nsamples, k=k, seed=123)

    assert isinstance(cvinds, list)
    assert len(cvinds) == k

    all_indices = np.concatenate(cvinds)
    assert set(all_indices) == set(range(nsamples))
    assert len(all_indices) == nsamples

    assert cvassigns.shape == (nsamples,)
    assert set(cvassigns) <= set(range(k))

    for fold_idx, inds in enumerate(cvinds):
        for i in inds:
            assert cvassigns[i] == fold_idx


def test_split_cfold_different_seed_changes_assignment():
    cvinds1, assign1 = split_cfold(12, k=4, seed=0)
    cvinds2, assign2 = split_cfold(12, k=4, seed=1)
    assert not np.array_equal(assign1, assign2)


def test_split_gfold_with_groupkfold(tmp_path, caplog):
    groups = np.array([0, 0, 1, 1])
    cv = GroupKFold(n_splits=2)

    cvinds, cvassigns = split_gfold(groups, cv)

    assert isinstance(cvinds, list)
    assert len(cvinds) == 2

    lengths = [len(fold) for fold in cvinds]
    assert sorted(lengths) == [2, 2]

    assert cvassigns.shape == groups.shape
    assert set(cvassigns) == set(range(len(cvinds)))

    fold_group_labels = [ {groups[i] for i in fold} for fold in cvinds ]
    for grp_set in fold_group_labels:
        assert len(grp_set) == 1

    combined = set().union(*fold_group_labels)
    assert combined == set(np.unique(groups))


def test_classification_validation_scores_perfect_prediction():
    ys = np.array([0, 1])
    eys = np.array([0, 1])
    ws = np.array([1.0, 1.0])
    pys = np.array([[0.9, 0.1], [0.1, 0.9]])

    scores = classification_validation_scores(ys, eys, ws, pys)

    expected_keys = {'accuracy', 'log_loss', 'auc', 'mean_confusion', 'mean_confusion_normalized'}
    assert expected_keys.issubset(set(scores.keys()))

    assert pytest.approx(scores['accuracy'], rel=1e-6) == 1.0
    assert pytest.approx(scores['auc'], rel=1e-6) == 1.0

    raw_cm = scores['mean_confusion']
    norm_cm = scores['mean_confusion_normalized']

    assert raw_cm == [[1, 0], [0, 1]]
    assert norm_cm == [[0.5, 0.0], [0.0, 0.5]]


def test_classification_validation_scores_misaligned():
    ys = np.array([0, 1])
    eys = np.array([0, 0])
    ws = np.array([1.0, 1.0])
    pys = np.array([[0.8, 0.2], [0.6, 0.4]])

    scores = classification_validation_scores(ys, eys, ws, pys)

    assert pytest.approx(scores['accuracy'], rel=1e-6) == 0.5
    assert 0.5 <= scores['auc'] <= 1.0
    assert scores['mean_confusion'] == [[1, 0], [1, 0]]


def test_regression_validation_scores_perfect_prediction():
    y_true = np.array([2.0, -1.0])
    y_pred = np.array([[2.0], [-1.0]])
    ws = np.array([1.0, 1.0])
    model = DummyRegressionModel()

    scores = regression_validation_scores(y_true, y_pred, ws, model)

    assert 'r2_score' in scores
    assert 'expvar' in scores

    assert pytest.approx(scores['r2_score'], rel=1e-6) == 1.0
    assert pytest.approx(scores['expvar'], rel=1e-6) == 1.0

    assert 'mse' in scores
    assert pytest.approx(scores['mse'], abs=1e-8) == 0.0


def test_regression_validation_scores_constant_prediction():
    y_true = np.array([1.0, 3.0])
    y_pred = np.array([[2.0], [2.0]])
    ws = np.array([1.0, 1.0])
    model = DummyRegressionModel()

    scores = regression_validation_scores(y_true, y_pred, ws, model)

    assert pytest.approx(scores['r2_score'], rel=1e-6) == 0.0
    assert pytest.approx(scores['mse'], rel=1e-6) == 1.0


def test_regression_validation_scores_transformed_model(tmp_path):
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.vstack([y_true,]).reshape(-1, 1)
    ws = np.ones_like(y_true)
    model = DummyRegressionModel()

    scores = regression_validation_scores(y_true, y_pred, ws, model)
    assert 'mll' not in scores
    assert 'lins_ccc' in scores
    assert 'smse' in scores


def test_join_dicts():
    input_dicts = [
        {'a': 1, 'b': 2},
        {'c': 3, 'd': 4}
    ]
    expected = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    result = _join_dicts(input_dicts)
    assert result == expected


def test_setup_validation_data_returns_cleaned_and_split_data():
    targets = DummyTargets()
    data = np.array([[1.0, 2.0],
                     [np.nan, np.nan],
                     [3.0, 4.0],
                     [5.0, 6.0]])
    mask = np.isnan(data)
    X = np.ma.masked_array(data, mask=mask)
    cleaned_X, y, lon_lat, groups, w, cv = setup_validation_data(X, targets, cv_folds=2, random_state=42)
    assert cleaned_X.shape[0] == 3
    assert y.shape == (3,)
    assert lon_lat.shape == (3, 2)
    assert groups.shape == (3,)
    assert w.shape == (3,)
    assert isinstance(cv, GroupKFold)
