import pickle

import numpy as np
import pytest
from sklearn.metrics import r2_score

from uncoverml.krige import krige_methods, Krige, all_ml_models, MLKrige
from uncoverml.models import regressors, classifiers
from uncoverml.optimise.models import transformed_modelmaps

modelmaps = {**regressors, **classifiers}
models = list(modelmaps.keys()) + list(transformed_modelmaps.keys())


@pytest.fixture(params=[k for k in models])
def get_models(request):
    if request.param in modelmaps:
        return modelmaps[request.param]
    elif request.param in transformed_modelmaps:
        return transformed_modelmaps[request.param]


def test_modeltags(get_models):

    model = get_models()
    tags = model.get_predict_tags()

    assert len(tags) >= 1  # at least a predict function

    if hasattr(model, 'predict_dist'):
        assert len(tags) >= 4  # at least predict, var and upper & lower quant

        if hasattr(model, 'entropy_reduction'):
            assert len(tags) == 5

        if hasattr(model, 'krige_residual'):
            assert len(tags) == 5

    else:
        if hasattr(model, 'entropy_reduction'):
            assert len(tags) == 2


def test_modelmap(get_models):

    mod = get_models()
    assert hasattr(mod, 'fit')
    assert hasattr(mod, 'predict')


def test_modelpickle(get_models):

    mod = get_models()
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
    make sure all optimise.models classes have ml_score attr
    """
    assert np.all([hasattr(get_transformed_model(), a) for a in
                   ['ml_score', 'score', 'fit', 'predict']])


@pytest.fixture(params=[k for k in all_ml_models
                        if k not in ['randomforest',
                                     'multirandomforest',
                                     'depthregress',
                                     'cubist',
                                     'multicubist',
                                     'decisiontree',
                                     'extratree'
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
