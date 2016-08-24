import pytest
from sklearn.metrics import r2_score

from uncoverml.models import modelmaps


@pytest.fixture(params=[k for k in modelmaps.keys()
                        if k not in ['depthregress', 'cubist']])
def get_models(request):
    return modelmaps[request.param]


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
