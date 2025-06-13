import os
import yaml
import joblib
import numpy as np
import pytest
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import uncoverml.scripts.superlearn_cli as sl
from sklearn.linear_model import LinearRegression


class DummyModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def fit(self, X, y):
        self.is_fitted = True
    def predict(self, X):
        return np.zeros(len(X))


@pytest.fixture(autouse=True)
def patch_modelmaps_and_dependencies(monkeypatch, tmp_path):
    monkeypatch.setitem(sl.all_modelmaps, "dummy_alg", DummyModel)

    class DummyWriter:
        def __init__(self, shape, bbox, crs, tif_name, partitions, outdir, band_tags):
            self.tif_name = tif_name
            self.written = []
        def write(self, arr, idx):
            self.written.append((arr.copy(), idx))
        def close(self):
            pass

    monkeypatch.setattr(sl, "ImageWriter", DummyWriter)
    monkeypatch.setattr(sl, "get_image_spec", lambda model, config: ((10, 10), None, None))

    class DummySource:
        def __init__(self, tif):
            pass

    monkeypatch.setattr(sl, "RasterioImageSource", DummySource)
    monkeypatch.setattr(sl, "extract_subchunks", lambda img, a, b, c: np.arange(10).reshape(-1, 1, 1, 1))
    monkeypatch.setattr(sl, "stacking", lambda models, x_mtrain, y_mtrain, x_all, **kw: (x_mtrain + 1, x_all + 2))
    class DummyStackTransformer:
        def __init__(self, models, regression, variant, metric, n_folds, shuffle, random_state, verbose):
            pass
        def fit(self, X, y, sample_weight=None):
            return self
        def transform(self, X):
            return X * 2
    monkeypatch.setattr(sl, "StackingTransformer", DummyStackTransformer)
    class DummySuperLearner:
        def __init__(self, scorer, folds, shuffle, sample_size):
            pass
        def add(self, models):
            pass
        def add_meta(self, model):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(X.shape[0])
    monkeypatch.setattr(sl, "SuperLearner", DummySuperLearner)
    yield


@pytest.fixture
def dummy_data():
    targets = SimpleNamespace(values=np.array([1, 2, 3]))
    x_shp = np.ma.masked_array(np.random.rand(3, 2), mask=[[0, 0], [0, 0], [0, 0]])
    return targets, x_shp


def test__grp_success_and_keyerror():
    d = {"a": "valueA"}
    assert sl._grp(d, "a") == "valueA"
    with pytest.raises(KeyError):
        sl._grp(d, "missing_key")


def test_define_model_and_load_model(tmp_path):
    yaml_content = {
        "learning": {"algorithm": "dummy_alg", "arguments": {"param1": 5}},
        "output": {"directory": str(tmp_path)}
    }
    yaml_path = tmp_path / "dummy_alg.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    model = sl.define_model(str(yaml_path))
    assert hasattr(model, "kwargs")
    assert model.kwargs["param1"] == 5
    model.kwargs["trained"] = True
    model_file = tmp_path / "dum.model"
    joblib.dump({"model": model, "config": str(yaml_path)}, str(model_file))
    loaded = sl.load_model(str(model_file))
    assert isinstance(loaded, dict)
    assert loaded["model"].kwargs["trained"] is True
    assert loaded["config"] == str(yaml_path)


def test_base_yaml(tmp_path):
    learn_alg_lst = [
        {"algorithm": "algA", "arguments": {}},
        {"algorithm": "algB", "arguments": {}}
    ]
    config_dic = {
        "output": {"directory": str(tmp_path)},
        "pickling": {"covariates": "", "targets": ""}
    }
    yaml_dic = sl.base_yaml(learn_alg_lst, config_dic)
    assert "algA" in yaml_dic and "algB" in yaml_dic
    for alg, yfile in yaml_dic.items():
        assert os.path.exists(yfile)
        with open(yfile) as f:
            data = yaml.safe_load(f)
            assert "learning" in data
            assert data["learning"]["algorithm"] == alg


def test_MetaLearn_and_meta_predict_and_meta_tif(tmp_path):
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    ml = sl.MetaLearn(LinearRegression)
    ml.fit(X, y)
    y_pred = ml.predict(X)
    assert y_pred.shape == (3,)
    mp = sl.meta_predict(ml, X)
    assert np.allclose(mp, y_pred)
    fake_model_conf = {"model": ml, "config": None}
    writer = sl.meta_tif(fake_model_conf, tif_name="Test", partitions=1)
    arr = np.zeros((10, 1))
    writer.write(arr, 0)
    writer.close()
    assert writer.tif_name == "Test"


def test_read_tifs_and_get_metafeats(tmp_path):
    tif_list = [str(tmp_path / f"f{i}.tif") for i in range(3)]
    for tif in tif_list:
        Path(tif).write_text("dummy")
    x_all = sl.read_tifs(tif_list)
    assert isinstance(x_all, np.ndarray) or isinstance(x_all, np.ma.MaskedArray)
    assert x_all.shape == (10, 3)
    x_meta = sl.get_metafeats(["alg1", "alg2"])
    assert isinstance(x_meta, np.ndarray) or isinstance(x_meta, np.ma.MaskedArray)
    assert x_meta.shape == (10, 2)


def test_v_stack_skstack_m_stack():
    x_mtrain = np.arange(8).reshape(4, 2)
    y_mtrain = np.array([0, 1, 2, 3])
    x_all = np.arange(12).reshape(6, 2)
    models = [LinearRegression(), LinearRegression()]
    meta_learner = LinearRegression
    y_all = sl.v_stack(x_mtrain, y_mtrain, x_all, models, meta_learner)
    assert isinstance(y_all, np.ndarray)
    assert y_all.shape[0] == x_all.shape[0]
    stack, meta_model = sl.skstack(
        x_mtrain, y_mtrain,
        [("m1", LinearRegression()), ("m2", LinearRegression())],
        meta_learner
    )
    assert hasattr(stack, "transform")
    assert hasattr(meta_model, "predict")
    ens = sl.m_stack(x_mtrain, y_mtrain, models, meta_learner)
    assert hasattr(ens, "fit")
    assert hasattr(ens, "predict")


def test_base_fit(monkeypatch, tmp_path):
    learn_alg_lst = [{'algorithm': 'dummy_alg'}]
    base_learner_lst = []
    model_lst = []
    config_dic = {'output': {'directory': str(tmp_path)}}
    targets = SimpleNamespace(values=np.array([1, 2, 3]), positions=np.array([[0, 0], [1, 1], [2, 2]]))
    x_shp = np.ma.masked_array(np.random.rand(3, 2), mask=np.zeros((3, 2)))
    dataset_dic = {'dummy_alg': (targets, x_shp)}
    dummy_yaml_path = tmp_path / 'dummy_alg.yml'
    dummy_yaml_path.write_text('learning:\n  algorithm: dummy_alg\narguments: {}')
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.Path.exists', lambda self: True)
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.Path.stat', lambda self: SimpleNamespace(st_size=1))
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.Config', lambda path: SimpleNamespace(
        algorithm='dummy_alg',
        n_subchunks=1,
        cross_validate=False,
        permutation_importance=False
    ))
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.define_model', lambda path: 'mock_model')
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.learn', lambda config, targets, x: None)
    dummy_model = {'model': 'trained_model', 'config': 'dummy_config'}
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.uncoverml.mpiops.run_once', lambda func, *args: dummy_model)
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.validate', lambda config, model, part: None)
    sl.base_fit(learn_alg_lst, base_learner_lst, model_lst, config_dic, dataset_dic, partitions=1)
    assert base_learner_lst == ['mock_model']
    assert model_lst[0]['model'] == 'trained_model'
    assert model_lst[0]['mfile'] == 'dummy_alg.model'


def test_validate_function(monkeypatch):
    model = SimpleNamespace()
    config = SimpleNamespace(
        pickle_load=True,
        oos_validation_file='dummy.shp',
        oos_validation_property='target',
    )
    dummy_targets = SimpleNamespace(values=np.array([1, 2, 3]))
    dummy_x = np.ma.masked_array(np.random.rand(3, 2), mask=[[0, 0], [0, 0], [0, 0]])
    monkeypatch.setattr(
        sl.uncoverml.scripts.uncoverml,
        '_load_data',
        lambda cfg, p: (dummy_targets, dummy_x)
    )
    monkeypatch.setattr(
        sl.uncoverml.validate,
        'oos_validate',
        lambda targets, x, model, cfg: None
    )
    sl.validate(config, model, partitions=1)
    assert config.pickle_load is False
    assert config.target_file == config.oos_validation_file
    assert config.target_property == config.oos_validation_property


def test_learn_function(monkeypatch, dummy_data):
    targets_shp, x_shp = dummy_data
    config = SimpleNamespace(
        algorithm='dummy_alg',
        cross_validate=True,
        permutation_importance=True,
        output=None
    )
    monkeypatch.setattr(sl.uncoverml.validate, 'local_crossval', lambda x, y, c: 'crossval_result')
    monkeypatch.setattr(sl.uncoverml.geoio, 'export_crossval', lambda result, conf: None)
    monkeypatch.setattr(sl.uncoverml.learn, 'local_learn_model', lambda x, y, c: 'trained_model')
    monkeypatch.setattr(sl.uncoverml.geoio, 'export_model', lambda model, conf: None)
    monkeypatch.setattr(sl.uncoverml.validate, 'permutation_importance', lambda m, x, y, c: None)
    monkeypatch.setattr(sl.uncoverml.mpiops, 'run_once', lambda fn, *args, **kwargs: fn(*args, **kwargs))

    sl.learn(config, targets_shp, x_shp)

def test_base_predict(tmp_path, monkeypatch):
    dummy_mask = tmp_path / 'dummy_mask.tif'
    dummy_mask.write_text('dummy')
    config_dic = {
        'mask': {'file': str(dummy_mask), 'retain': 1},
        'validation': True
    }
    alg_name = 'dummy_alg'
    learn_alg_lst = [{'algorithm': alg_name}]
    outdir = tmp_path / f'{alg_name}_out'
    outdir.mkdir()
    results_file = outdir / f'{alg_name}_results.csv'
    df = pd.DataFrame({
        'y_pred': [1, 2],
        'y_true': [1, 2],
        'lat': [-10.0, -11.0],
        'lon': [130.0, 131.0],
        'y_transformed': [1, 2]
    })
    df.to_csv(results_file, index=False)
    model_lst = [{
        'model': MagicMock(),
        'config': None,
        'outdir': str(outdir),
        'mfile': f'{alg_name}.model'
    }]
    monkeypatch.setattr(
        'uncoverml.scripts.uncoverml.predict.callback',
        lambda model_file, partitions, mask, retain: None
    )
    df_pred = sl.base_predict(learn_alg_lst, model_lst, config_dic, partitions=1)
    assert isinstance(df_pred, pd.DataFrame)
    assert df_pred.shape[0] == 2


def test_combine_pred(monkeypatch):
    learn_alg_lst = [{'algorithm': 'alg1'}, {'algorithm': 'alg2'}]
    base_learner_lst = ['m1', 'm2']
    model_lst = [{'model': 'm1'}, {'model': 'm2'}]
    df_pred = pd.DataFrame({
        'y_true': [1, 2],
        'alg1': [0.1, np.nan],
        'alg2': [0.3, 0.4]
    })
    x_tif_mock = np.ma.array([[0.1, 0.3], [np.nan, 0.4]])
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.get_metafeats', lambda algs: x_tif_mock)
    x_tif = sl.combine_pred(learn_alg_lst, base_learner_lst, model_lst, df_pred)
    assert x_tif.shape[1] == 1
    assert learn_alg_lst == [{'algorithm': 'alg2'}]
    assert model_lst == [{'model': 'm2'}]
    assert base_learner_lst == ['m2']

def test_go_vecml(monkeypatch):
    learn_alg_lst = [{'algorithm': 'alg1'}, {'algorithm': 'alg2'}]
    base_learner_lst = ['model1', 'model2']
    model_conf = {
    'model': 'dummy',
    'config': type('Config', (), {'n_subchunks': 4})(),
    'outdir': '/tmp',
    'mfile': 'dummy.model'
    }
    targets_shp = type('Targets', (), {
        'observations': np.array([1.0, 2.0]),
        'weights': np.array([1.0, 1.0])
    })()
    x_shp = np.ma.array([[0.1, 0.2], [0.3, 0.4]])
    class DummyWriter:
        def __init__(self, *args, **kwargs): self.data = []
        def write(self, arr, idx): self.data.append(arr)
        def close(self): pass
    monkeypatch.setattr(sl, 'meta_tif', lambda *a, **k: DummyWriter())
    class DummyStack:
        def transform(self, x): return x * 2
    class DummyModel:
        def predict(self, x): return np.array([5.0, 6.0])
    monkeypatch.setattr(sl, 'skstack', lambda *a, **k: (DummyStack(), DummyModel()))
    monkeypatch.setattr(sl, 'm_stack', lambda *a, **k: DummyModel())
    monkeypatch.setattr(sl, '_get_data', lambda part, conf: (np.array([[1.0, 2.0], [3.0, 4.0]]), None))
    monkeypatch.setattr(sl.uncoverml.mpiops, 'run_once', lambda f, *a, **k: f(*a, **k))
    sl.go_vecml(learn_alg_lst, base_learner_lst, model_conf, targets_shp, x_shp, LinearRegression)


def test_go_meta_runs(monkeypatch, tmp_path):
    df_pred = pd.DataFrame({
        'y_true': [1, 2],
        'alg1': [0.1, 0.2],
        'alg2': [0.3, 0.4],
        'lat': [-10, -11],
        'lon': [130, 131]
    })
    x_tif = np.ma.array([[0.1, 0.3], [0.2, 0.4]])
    model_conf = [{
        'model': LinearRegression(),
        'config': None,
        'outdir': str(tmp_path),
        'mfile': 'dummy.model'
    }]
    config_dic = {
        'validation': [[{'k-fold': {'folds': 3}}]]
    }
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.meta_cv', lambda *args, **kwargs: None)
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.meta_fit', lambda *a, **kw: LinearRegression())
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.meta_predict', lambda model, X: np.array([1.1, 1.2]))
    class DummyWriter:
        def __init__(self, *a, **k): self.arr = []
        def write(self, arr, idx): self.arr.append(arr)
        def close(self): pass
    monkeypatch.setattr('uncoverml.scripts.superlearn_cli.meta_tif', lambda *a, **k: DummyWriter())
    sl.go_meta(df_pred, x_tif, model_conf, config_dic, LinearRegression)


def test_meta_cv_runs(monkeypatch):
    X = np.ma.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.ma.array([1, 2, 3, 4, 5])
    class DummyMeta:
        def __init__(self, model, **kwargs):
            self.model = LinearRegression()
        def fit(self, X, y): return self
        def predict(self, X): return np.sum(X, axis=1)

    monkeypatch.setattr(sl, 'MetaLearn', DummyMeta)
    monkeypatch.setattr(sl, 'regression_validation_scores', lambda y_true, y_pred, w, m: {'r2': 1.0})
    monkeypatch.setattr(sl, 'r2_score', lambda y_true, y_pred: 0.95)
    monkeypatch.setattr(sl, 'explained_variance_score', lambda y_true, y_pred: 0.9)
    monkeypatch.setattr(sl, 'smse', lambda y_true, y_pred: 0.05)
    monkeypatch.setattr(sl, 'lins_ccc', lambda y_true, y_pred: 0.92)
    monkeypatch.setattr(sl, 'mean_squared_error', lambda y_true, y_pred: 0.1)
    monkeypatch.setattr(sl.plt, 'figure', lambda: None)
    monkeypatch.setattr(sl.plt, 'scatter', lambda x, y: None)
    monkeypatch.setattr(sl.plt, 'xlabel', lambda l: None)
    monkeypatch.setattr(sl.plt, 'ylabel', lambda l: None)
    monkeypatch.setattr(sl.plt, 'savefig', lambda fn: None)
    monkeypatch.setattr(sl.plt, 'close', lambda: None)

    sl.meta_cv(X, y, LinearRegression, n_splits=2)
