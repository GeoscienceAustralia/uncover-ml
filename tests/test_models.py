import pytest
from sklearn.metrics import r2_score
import numpy as np

from uncoverml.models import modelmaps
from uncoverml.optimise.models import transformed_modelmaps
from uncoverml.krige.krige import krige_methods, Krige

models = list(modelmaps.keys()) + list(transformed_modelmaps.keys())


@pytest.fixture(params=[k for k in models
                        if k not in ['depthregress',
                                     'cubist',
                                     'multicubist',
                                     'multirandomforest']])
def get_models(request):
    if request.param in modelmaps:
        return modelmaps[request.param]
    elif request.param in transformed_modelmaps:
        return transformed_modelmaps[request.param]


def test_modeltags(get_models):

    model = get_models()
    tags = model.get_predict_tags()

    assert len(tags) >= 1  # at least a predict function

    if hasattr(model, 'predict_proba'):
        assert len(tags) >= 4  # at least predict, var and upper & lower quant

        if hasattr(model, 'entropy_reduction'):
            assert len(tags) == 5

    else:
        if hasattr(model, 'entropy_reduction'):
            assert len(tags) == 2


def test_modelmap(linear_data, get_models):

    yt, Xt, ys, Xs = linear_data
    mod = get_models()

    mod.fit(Xt, yt)
    Ey = mod.predict(Xs)

    assert r2_score(ys, Ey) > 0


@pytest.fixture(params=krige_methods.keys())
def get_krige_method(request):
    return request.param


def test_krige(linear_data, get_krige_method):

    yt, Xt, ys, Xs = linear_data

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


def test_mlkrige():
    """
    placeholder
    """
    pass


# def test_modelpersistance(make_fakedata):

#     X, y, _, mod_dir = make_fakedata

#     for model in models.modelmaps.keys():
#         mod = models.modelmaps[model]()
#         mod.fit(X, y)

#         with open(path.join(mod_dir, model + ".pk"), 'wb') as f:
#             pickle.dump(mod, f)

#         with open(path.join(mod_dir, model + ".pk"), 'rb') as f:
#             pmod = pickle.load(f)

#         Ey = pmod.predict(X)

#         assert Ey.shape == y.shape
