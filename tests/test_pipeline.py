import pytest

from uncoverml import pipeline, models


@pytest.fixture(params=[k for k in models.modelmaps.keys()
                        if k != 'depthregress'])
def get_modelnames(request):
    return request.param


# def test_learn_predict(linear_data, get_modelnames):

#     yt, Xt, ys, Xs = linear_data

#     class targets:
#         observations = yt
#         fields = {}

#     mod = pipeline.local_learn_model(Xt, targets, get_modelnames, {})
#     predictions = pipeline.predict(Xs, mod)

#     assert len(predictions) == len(ys)
#     assert predictions.shape[1] == len(mod.get_predict_tags())


# TODO Test cross_validate
