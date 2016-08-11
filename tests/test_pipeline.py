import pytest

from uncoverml import pipeline, models


@pytest.fixture(params=list(models.modelmaps.keys()))
def get_modelnames(request):
    return request.param


def test_learn_predict(linear_data, get_modelnames):

    yt, Xt, ys, Xs = linear_data

    class targets:
        observations = yt

    mod = pipeline.learn_model(Xt, targets, get_modelnames, {})
    predictions = pipeline.predict(Xs, mod)

    assert len(predictions) == len(ys)
    assert predictions.shape[1] == len(mod.get_predict_tags())
