import pickle
import numpy as np
import pytest
from sklearn.metrics import r2_score

from uncoverml.krige import krige_methods, Krige, all_ml_models, MLKrige
from uncoverml.models import (apply_masked,
                              apply_multiple_masked,
                              modelmaps)
from uncoverml.optimise.models import transformed_modelmaps, XGBQuantileWrapper

models = {**transformed_modelmaps, **modelmaps}


@pytest.fixture(params=[v for v in models.values()])
def get_models(request):
    return request.param


def test_modeltags(get_models):

    model = get_models()

    # Patch classifiers since they only get their tags when "fit" called
    if hasattr(model, 'predict_proba'):
        model.le.classes_ = ('a', 'b', 'c')

    tags = model.get_predict_tags()

    assert len(tags) >= 1  # at least a predict function for regression

    if hasattr(model, 'predict_dist'):
        assert len(tags) >= 4  # at least predict, var and upper & lower quant

        if hasattr(model, 'entropy_reduction'):
            assert len(tags) == 5

        if hasattr(model, 'krige_residual'):
            assert len(tags) == 5

    elif hasattr(model, 'predict_proba'):
        assert len(tags) == 4
        assert tags == ['most_likely', 'a_0', 'b_1', 'c_2']


def test_modelmap(get_models):

    mod = get_models()

    assert hasattr(mod, 'fit')
    assert hasattr(mod, 'predict')


def test_modelpickle(get_models):
    mod = get_models()

    # don't test picking for multirandomforest as pickling is handled by
    # the individual randomforest jobs downstream
    if isinstance(mod, models['multirandomforest']):
        return
    mod_str = pickle.dumps(mod)
    mod_pic = pickle.loads(mod_str)

    # Make sure all the keys survive the pickle, even if the objects differ
    assert mod.__dict__.keys() == mod_pic.__dict__.keys()


@pytest.fixture(params=krige_methods.keys())
def get_krige_method(request):
    return request.param


def test_krige(linear_data, get_krige_method):

    yt, Xt, ys, Xs = linear_data()

    mod = Krige(method=get_krige_method)
    mod.fit(np.tile(Xt, (1, 2)), yt)
    Ey = mod.predict(np.tile(Xs, (1, 2)))
    assert r2_score(ys, Ey) > 0


@pytest.fixture(params=[k for k in transformed_modelmaps])
def get_transformed_model(request):
    return transformed_modelmaps[request.param]


def test_trasnsformed_model_attr(get_transformed_model):
    """
    make sure all optimise.models classes have the required attributes
    """
    if isinstance(get_transformed_model(), XGBQuantileWrapper):
        pytest.skip(msg='xgbquantile is just a wrapper')
    assert np.all([hasattr(get_transformed_model(), a) for a in
                   ['score', 'fit', 'predict']])


@pytest.fixture(params=[k for k in all_ml_models
                        if k not in ['randomforest',
                                      'multirandomforest',
                                      'depthregress',
                                      'cubist',
                                      'multicubist',
                                      'decisiontree',
                                      'extratree',
                                      'xgbquantile',
                                      # 'catboost'
                                     ]])
def models_supported(request):
    return request.param


def test_mlkrige(models_supported, get_krige_method):
    """
    tests algos that can be used with MLKrige
    """
    mlk = MLKrige(ml_method=models_supported, method=get_krige_method)
    assert hasattr(mlk, 'fit')
    assert hasattr(mlk, 'predict')

    mod_str = pickle.dumps(mlk)
    mod_pic = pickle.loads(mod_str)
    # Make sure all the keys survive the pickle, even if the objects differ
    assert mlk.__dict__.keys() == mod_pic.__dict__.keys()



def test_apply_masked(masked_data):
    yt, Xt, ys, Xs = masked_data
    yt_masked = np.ma.masked_array(yt, mask=Xt.mask.flatten())

    def ident(X):
        return X

    def target(X):
        return yt[~Xt.mask.flatten()]

    def fit(X):
        assert np.allclose(X, Xt.data[~Xt.mask.flatten()])
        return

    assert np.ma.all(Xt == apply_masked(ident, Xt))
    assert np.ma.all(yt_masked == apply_masked(target, Xt))
    assert apply_masked(fit, Xt) is None


def test_apply_multiple_masked(masked_data):
    yt, Xt, ys, Xs = masked_data
    yt_masked = np.ma.masked_array(yt, mask=Xt.mask.flatten())

    def fit(X, y):
        assert np.allclose(X, Xt.data[~Xt.mask.flatten()])
        assert np.allclose(y, yt_masked.data[~yt_masked.mask.flatten()])
        return

    def predict(X, y):
        return y

    yr = apply_multiple_masked(predict, (Xt, yt_masked))
    assert np.ma.all(yt_masked == yr)
    assert apply_multiple_masked(fit, (Xt, yt_masked)) is None
