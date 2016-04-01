import numpy as np

from uncoverml import models


def test_ModelSpec(make_fakedata):

    X, y, w = make_fakedata

    mspec = models.ModelSpec('sklearn.ensemble', 'RandomForestRegressor')
    # import IPython; IPython.embed(); exit()
    mspec = models.learn_model(X, y, mspec)
    Ey = models.predict_model(X, mspec)

    assert np.allclose(Ey, y)
