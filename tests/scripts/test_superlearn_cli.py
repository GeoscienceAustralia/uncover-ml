import os
import yaml
import joblib
import numpy as np
import pytest
from pathlib import Path

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
