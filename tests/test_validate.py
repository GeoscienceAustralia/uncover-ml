import os
import tempfile
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sklearn.model_selection import GroupKFold, KFold
from uncoverml.validate import (
    _join_dicts,
    split_cfold,
    split_gfold,
    classification_validation_scores,
    regression_validation_scores,
    setup_validation_data,
    permutation_importance,
    local_rank_features,
    local_crossval,
    CrossvalInfo,
    plot_feature_importance,
    oos_validate,
    plot_feature_correlation_matrix,
    validation_scatter,
    residual_plot,
    plot_permutation_feature_importance
)


class DummyIdentityTransform:
    def transform(self, x):
        return x


class DummyRegressionModel:
    def __init__(self):
        self.target_transform = DummyIdentityTransform()

    def get_predict_tags(self):
        return ["Prediction"]

    def fit(self, x, y, **kwargs):
        return self


class DummyTargets:
    def __init__(self):
        self.observations = np.array([1.0, 2.0, 3.0, 4.0])
        self.weights = np.array([1.0, 1.0, 1.0, 1.0])
        self.positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.groups = np.array([0, 0, 1, 1])
        self.fields = {
            'soil_type': np.array([0, 1, 0, 1])
        }


class DummyCfg:
    def __init__(self, algorithm, output_dir, name, final_transform):
        self.algorithm = algorithm
        self.output_dir = output_dir
        self.name = name
        self.final_transform = final_transform
        self.cubist = False              
        self.multicubist = False
        self.multirandomforest = False
        self.multitarget = False
        self.rank_features = False
        self.save_predictions = False
        self.parallel_validate = False


class DummyScore:
    def __init__(self):
        self.scores = {'r2_score': 0.8, 'mse': 0.2}


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


@patch("uncoverml.validate.eli5.explain_weights_df", autospec=True)
@patch("uncoverml.validate.geoio.feature_names", autospec=True)
@patch("uncoverml.validate.apply_multiple_masked", autospec=True)
@patch("uncoverml.validate.PermutationImportance", autospec=True)
@patch("uncoverml.validate.transformed_modelmaps", new_callable=lambda: {"DummyAlg": object()})
def test_permutation_importance(mock_modelmaps, mock_PI, mock_apply, mock_feature_names, mock_eli5, tmp_path):
    class TestPI:
        def __init__(self, model, scoring, cv, n_iter, refit):
            self.model = model
            self.scoring = scoring
        def fit(self, X, y):
            self.feature_importances_ = np.arange(X.shape[1]) + 0.1
            return self
    mock_PI.side_effect = TestPI
    mock_apply.side_effect = lambda func, data, **kw: func(*data)
    mock_feature_names.side_effect = lambda conf: [f"f{i}" for i in range(3)]
    mock_eli5.side_effect = lambda pi_cv, feature_names, top: pd.DataFrame({"feature": feature_names, "weight": np.ones(len(feature_names))})
    x_all = np.random.RandomState(0).randn(6, 3)
    targets_all = DummyTargets()
    cfg = DummyCfg(algorithm="DummyAlg", output_dir=str(tmp_path), name="permunit", final_transform= "dummy_transform" )
    model = DummyRegressionModel()
    permutation_importance(model, x_all, targets_all, cfg)
    expected = ["explained_variance", "r2", "neg_mean_absolute_error", "neg_mean_squared_error"]
    for s in expected:
        p = tmp_path / f"permunit_permutation_importance_{s}.csv"
        assert p.exists(), f"Missing CSV for score '{s}'"
        df = pd.read_csv(p)
        assert set(df.columns) == {"feature", "weight"}
        assert len(df) == 3


@patch("uncoverml.validate.transformed_modelmaps", new_callable=lambda: {"Allowed": object()})
def test_permutation_importance_raises_exception(mock_modelmaps, tmp_path):
    x_all = np.random.RandomState(1).randn(4, 2)
    targets_all = DummyTargets()
    cfg = DummyCfg(algorithm="NotAllowed", output_dir=str(tmp_path), name="bad", final_transform= "dummy_transform")
    model = DummyRegressionModel()
    with pytest.raises(AttributeError):
        permutation_importance(model, x_all, targets_all, cfg)


@patch("uncoverml.validate.mpiops.chunk_index", 0)
@patch("uncoverml.validate.local_crossval", autospec=True)
@patch("uncoverml.validate.targ.gather_targets_main", autospec=True)
@patch("uncoverml.validate.feat.gather_features", autospec=True)
@patch("uncoverml.validate.feat.transform_features", autospec=True)
def test_local_rank_features(mock_transform_features, mock_gather_features, mock_gather_targets, mock_crossval, tmp_path):
    image_chunk_sets = [
        {'f1.tif': 'data1', 'f2.tif': 'data2'},
        {'f2.tif': 'data3'}
    ]
    transform_sets = ['ts1', 'ts2']
    targets = MagicMock()
    config = DummyCfg("DummyAlg", str(tmp_path), "perftest", "dummy_transform")
    mock_transform_features.return_value = (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1]))
    mock_gather_features.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    mock_gather_targets.return_value = targets
    mock_crossval.return_value = DummyScore()
    measures, features, scores = local_rank_features(image_chunk_sets, transform_sets, targets, config)
    assert measures == ['r2_score', 'mse']
    assert features == ['f1', 'f2']
    assert scores.shape == (2, 2)


@patch("uncoverml.validate.mpiops.chunk_index", 0)
def test_local_rank_features_raises_exception(tmp_path):
    image_chunk_sets = [{'onlyone.tif': 'data'}]
    transform_sets = ['ts1']
    targets = MagicMock()
    config = DummyCfg(algorithm="NotAllowed", output_dir=str(tmp_path), name="bad", final_transform= "dummy_transform")
    with pytest.raises(ValueError, match="only one feature"):
        local_rank_features(image_chunk_sets, transform_sets, targets, config)


@patch("uncoverml.validate.mpiops.chunk_index", 1)
@patch("uncoverml.validate.local_crossval", return_value=DummyScore())
@patch("uncoverml.validate.targ.gather_targets_main", return_value=np.array([1.0, 2.0]))
@patch("uncoverml.validate.feat.gather_features", return_value=np.array([[1, 2], [3, 4]]))
@patch("uncoverml.validate.feat.transform_features",return_value=(np.array([[1, 2], [3, 4]]), [0, 1]))
def test_local_rank_features_returns_none(mock_transform, mock_gather, mock_targets, mock_crossval):
    image_chunk_sets = [
        {'f1.tif': 'data1', 'f2.tif': 'data2'},
        {'f2.tif': 'data3'}
    ]
    transform_sets = ['ts1', 'ts2']
    targets = MagicMock()
    config = DummyCfg("DummyAlg", "/tmp", "skip", "dummy_transform")
    result = local_rank_features(image_chunk_sets, transform_sets, targets, config)
    assert result == (None, None, None)


@patch("uncoverml.validate.modelmaps", new={"DummyAlg": DummyRegressionModel})
@patch("uncoverml.validate.mpiops.comm", new=MagicMock())
@patch("uncoverml.validate.mpiops.chunks", new=1)
@patch("uncoverml.validate.mpiops.chunk_index", new=0)
@patch("uncoverml.validate.split_gfold", autospec=True)
@patch("uncoverml.validate.setup_validation_data", autospec=True)
@patch("uncoverml.validate.apply_multiple_masked", autospec=True)
@patch("uncoverml.validate.predict.predict", autospec=True)
def test_local_crossval(mock_predict, mock_apply_multiple_masked, mock_setup_validation_data, mock_split_gfold):
    x_all = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    lon_lat = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    groups = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    cv = MagicMock()
    cv_indices = np.array([0, 0, 1, 1])
    targets = DummyTargets()
    cfg = DummyCfg("DummyAlg", "/tmp", "xvaltest", "dummy_transform")
    cfg.folds = 2
    cfg.algorithm_args = {}
    mock_setup_validation_data.return_value = (x_all, y, lon_lat, groups, weights, cv)
    mock_split_gfold.return_value = ([], cv_indices)
    mock_predict.return_value = np.array([[10.0], [30.0]])
    mock_apply_multiple_masked.side_effect = lambda func, data, **kwargs: func(*data)
    result = local_crossval(x_all, targets, cfg)
    assert isinstance(result, CrossvalInfo)
    assert result.y_true.shape[0] == 4
    assert "Prediction" in result.y_pred
    assert result.y_pred["Prediction"].shape[0] == 4
    assert isinstance(result.scores, dict)
    assert all(isinstance(v, float) for v in result.scores.values())


@patch.dict("uncoverml.validate.transformed_modelmaps", {"DummyAlg": object()})
@patch("uncoverml.validate.plot_permutation_feature_importance", autospec=True)
@patch("uncoverml.validate.write_progress_to_file", autospec=True)
@patch("uncoverml.validate.eli5.explain_weights_df", autospec=True)
@patch("uncoverml.validate.geoio.feature_names", return_value=["f1", "f2"])
@patch("uncoverml.validate.apply_multiple_masked", autospec=True)
@patch("uncoverml.validate.PermutationImportance", autospec=True)
def test_plot_feature_importance_regression_oos_false(mock_permimp, mock_apply_multiple_masked, mock_feature_names, mock_explain_weights_df, mock_write_progress, mock_plot_perm, tmp_path):
    x_all = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    targets = DummyTargets()
    cfg = DummyCfg("DummyAlg", str(tmp_path), "featimp", "dummy_transform")
    model = MagicMock(spec=["fit"])
    pi_mock = MagicMock()
    pi_mock.fit.return_value = pi_mock
    mock_permimp.return_value = pi_mock
    mock_apply_multiple_masked.side_effect = lambda func, data, **kwargs: func(*data)
    df_mock = pd.DataFrame({"feature": ["f1", "f2"], "weight": [0.6, 0.4]})
    mock_explain_weights_df.side_effect = [df_mock, df_mock, df_mock, df_mock]
    plot_feature_importance(model, x_all, targets, cfg, oos_val=False)
    assert mock_write_progress.call_count == 13
    assert mock_plot_perm.call_count == 4
    called_scores = [call.args[4] for call in mock_plot_perm.call_args_list]
    assert set(called_scores) == {"explained_variance", "r2", "neg_mean_absolute_error", "neg_mean_squared_error"}
    for score in ["explained_variance", "r2", "neg_mean_absolute_error", "neg_mean_squared_error"]:
        csv_path = tmp_path / f"featimp_permutation_importance_{score}.csv"
        assert csv_path.exists(), f"expected CSV not found: {csv_path}"
        df = pd.read_csv(csv_path)
        assert set(df.columns) == {"feature", "weight"}
        assert len(df) == 2


@patch.dict("uncoverml.validate.transformed_modelmaps", {"DummyAlg": object()})
@patch("uncoverml.validate.plot_permutation_feature_importance", autospec=True)
@patch("uncoverml.validate.write_progress_to_file", autospec=True)
@patch("uncoverml.validate.eli5.explain_weights_df", autospec=True)
@patch("uncoverml.validate.geoio.feature_names", return_value=["f1", "f2"])
@patch("uncoverml.validate.apply_multiple_masked", autospec=True)
@patch("uncoverml.validate.PermutationImportance", autospec=True)
def test_plot_feature_importance_regression_oos_true(mock_permimp, mock_apply_multiple_masked, mock_feature_names, mock_explain_weights_df,
mock_write_progress, mock_plot_perm, tmp_path):
    x_all = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0],
                      [7.0, 8.0]])
    targets = DummyTargets()
    cfg = DummyCfg("DummyAlg", str(tmp_path), "featimp", "dummy_transform")
    model = MagicMock(spec=["fit"])
    pi_mock = MagicMock()
    pi_mock.fit.return_value = pi_mock
    mock_permimp.return_value = pi_mock
    mock_apply_multiple_masked.side_effect = lambda func, data, **kwargs: func(*data)
    df_mock = pd.DataFrame({"feature": ["f1", "f2"], "weight": [0.5, 0.5]})
    mock_explain_weights_df.side_effect = [df_mock, df_mock, df_mock, df_mock]
    plot_feature_importance(model, x_all, targets, cfg, oos_val=True)
    assert mock_write_progress.call_count == 13
    assert mock_plot_perm.call_count == 4
    for score in ["explained_variance", "r2", "neg_mean_absolute_error", "neg_mean_squared_error"]:
        csv_path = tmp_path / f"featimp_permutation_importance_oos_{score}.csv"
        assert csv_path.exists(), f"expected CSV not found: {csv_path}"
        df = pd.read_csv(csv_path)
        assert set(df.columns) == {"feature", "weight"}
        assert len(df) == 2


@patch.dict("uncoverml.validate.transformed_modelmaps", {"Allowed": object()})
def test_plot_feature_importance_raises_exception(tmp_path):
    x_all = np.random.RandomState(0).randn(4, 2)
    targets = DummyTargets()
    cfg = DummyCfg("NotAllowed", str(tmp_path), "bad", "dummy_transform")
    model = MagicMock(spec=["fit"])
    with pytest.raises(AttributeError):
        plot_feature_importance(model, x_all, targets, cfg, oos_val=False)


@patch("uncoverml.validate.plot_feature_correlation_matrix", autospec=True)
@patch("uncoverml.validate.residual_plot", autospec=True)
@patch("uncoverml.validate.validation_scatter", autospec=True)
@patch("uncoverml.validate.geoio.output_json", autospec=True)
@patch("uncoverml.validate.regression_validation_scores", autospec=True)
@patch("uncoverml.validate.write_progress_to_file", autospec=True)
@patch("uncoverml.validate.predict.predict", autospec=True)
@patch("uncoverml.validate.mpiops.chunk_index", 0)
def test_oos_validate(mock_predict, mock_write_progress, mock_reg_scores, mock_output_json, mock_validation_scatter,
    mock_residual_plot, mock_feat_corr, tmp_path):
    x_all = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0],
                      [7.0, 8.0]])
    targets = DummyTargets()
    cfg = DummyCfg("DummyAlg", str(tmp_path), "oostest", "dummy_transform")
    model = DummyRegressionModel()
    case = [
        (False, None,
         np.array([[10.0], [20.0], [30.0], [40.0]]),
         "", ""),
        (True, (0.05, 0.95),
         np.array([[11.0], [21.0], [31.0], [41.0]]),
         "_oos", "_oos"),
    ]

    for oos_flag, quantiles, preds, csv_suffix, json_suffix in case:
        for m in (mock_predict, mock_write_progress, mock_reg_scores, mock_output_json, mock_validation_scatter,
                  mock_residual_plot, mock_feat_corr):
            m.reset_mock()

        cfg.quantiles = quantiles
        mock_predict.return_value = preds
        mock_reg_scores.return_value = {
            "r2_score": 0.99 if oos_flag else 1.0,
            "mse": 0.01 if oos_flag else 0.0,
            "lins_ccc": 0.99 if oos_flag else 1.0,
            "smse": 0.01 if oos_flag else 0.0,
        }
        oos_validate(targets, x_all, model, cfg, oos_validate=oos_flag)
        mock_predict.assert_called_once()
        _, kwargs = mock_predict.call_args
        assert kwargs.get("interval", None) is cfg.quantiles
        assert np.array_equal(kwargs.get("lon_lat"), targets.positions)
        csv_path = tmp_path / f"oostest_validation{csv_suffix}.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["Prediction", "y_true", "lon", "lat"]
        assert len(df) == preds.shape[0]
        assert mock_output_json.call_count == 1
        json_dest = mock_output_json.call_args[0][1]
        assert str(json_dest).endswith(f"oostest_validation_scores{json_suffix}.json")
        assert mock_write_progress.call_count == 8
        mock_validation_scatter.assert_called_once()
        assert mock_validation_scatter.call_args[0][0] is cfg
        assert mock_validation_scatter.call_args[0][3] is oos_flag
        mock_residual_plot.assert_called_once()
        r_args = mock_residual_plot.call_args[0]
        assert r_args[0] is cfg and r_args[1] is preds and np.array_equal(r_args[2], targets.observations)
        assert r_args[3] is oos_flag


@patch("uncoverml.validate.sns.heatmap", autospec=True)
@patch("uncoverml.validate.geoio.feature_names", return_value=["a.tif", "b.tif"])
def test_plot_feature_correlation_matrix(mock_feat_names, mock_heatmap, tmp_path):
    cfg = DummyCfg("DummyAlg", str(tmp_path), "corrtest", "dummy_transform")
    x_all = np.array([[1.0, 2.0],
                      [2.0, 3.0],
                      [3.0, 5.0]])
    plot_feature_correlation_matrix(cfg, x_all, oos_val=False)
    out = tmp_path / "corrtest_feature_correlation.png"
    assert out.exists()
    df = pd.read_csv(out) if out.suffix == ".csv" else None
    plot_feature_correlation_matrix(cfg, x_all, oos_val=True)
    out_oos = tmp_path / "corrtest_feature_correlation_oos.png"
    assert out_oos.exists()
    assert mock_heatmap.call_count == 2


def test_validation_scatter(tmp_path):
    cfg = DummyCfg("DummyAlg", str(tmp_path), "scattest", "dummy_transform")
    for suffix in ("", "_oos"):
        (tmp_path / f"scattest_validation_scores{suffix}.json").write_text(
            json.dumps({"r2_score": 0.9, "lins_ccc": 0.88, "mse": 0.11, "smse": 0.12})
        )
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    preds = np.array([1.1, 1.9, 3.1, 3.9])
    validation_scatter(cfg, y_true, preds, False)
    assert (tmp_path / "scattest_real_vs_pred_density_scatter.png").exists()
    validation_scatter(cfg, y_true, preds, True)
    assert (tmp_path / "scattest_real_vs_pred_density_scatter_oos.png").exists()


@patch("uncoverml.validate.sns.residplot", autospec=True)
def test_residual_plot(mock_residplot, tmp_path):
    cfg = DummyCfg("DummyAlg", str(tmp_path), "residtest", "dummy_transform")
    predictions = np.array([[1.0], [2.0], [3.0], [4.0]])
    observations = np.array([1.1, 1.8, 3.2, 3.9])
    residual_plot(cfg, predictions, observations, oos_val=False)
    out = tmp_path / "residtest_residuals.png"
    assert out.exists()
    residual_plot(cfg, predictions, observations, oos_val=True)
    out_oos = tmp_path / "residtest_residuals_oos.png"
    assert out_oos.exists()
    assert mock_residplot.call_count == 2


@patch.dict("uncoverml.validate.transformed_modelmaps", {"DummyAlg": object()})
@patch("uncoverml.validate.sns.barplot", autospec=True)
@patch("uncoverml.validate.eli5.explain_weights_df", autospec=True)
@patch("uncoverml.validate.geoio.feature_names", return_value=["f1.tif", "f2.tif"])
@patch("uncoverml.validate.apply_multiple_masked", autospec=True)
@patch("uncoverml.validate.PermutationImportance", autospec=True)
def test_plot_permutation_feature_importance(mock_PI, mock_apply, mock_feature_names, mock_eli5, mock_barplot, tmp_path):
    x_all = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
    targets = DummyTargets()
    cfg = DummyCfg("DummyAlg", str(tmp_path), "permtest", "dummy_transform")
    model = MagicMock(spec=["fit"])
    pi_mock = MagicMock()
    pi_mock.fit.return_value = pi_mock
    mock_PI.return_value = pi_mock
    mock_apply.side_effect = lambda func, data, **kw: func(*data)
    df = pd.DataFrame({"feature": ["f1", "f2"], "weight": [0.7, 0.3]})
    mock_eli5.return_value = df
    for oos_flag, csv_suffix, png_suffix in [
        (False, "permutation_importance_", "feature_importance_bars_"),
        (True,  "permutation_importance_oos_", "feature_importance_bars_oos_"),
    ]:
        plot_permutation_feature_importance(model, x_all, targets, cfg, score="r2", oos_val=oos_flag)
        csv_path = tmp_path / f"permtest_{csv_suffix}r2.csv"
        png_path = tmp_path / f"permtest_{png_suffix}r2.png"
