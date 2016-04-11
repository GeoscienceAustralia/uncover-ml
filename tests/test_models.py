import pickle
from os import path
from sklearn.metrics import r2_score

from uncoverml import models


def test_modelmap(make_fakedata):

    X, y, _, _ = make_fakedata

    for model in models.modelmaps.keys():
        mod = models.modelmaps[model]()
        mod.fit(X, y)

        Ey = mod.predict(X)

        assert r2_score(y, Ey) > 0


def test_modelpersistance(make_fakedata):

    X, y, _, mod_dir = make_fakedata

    for model in models.modelmaps.keys():
        mod = models.modelmaps[model]()
        mod.fit(X, y)

        with open(path.join(mod_dir, model + ".pk"), 'wb') as f:
            pickle.dump(mod, f)

        with open(path.join(mod_dir, model + ".pk"), 'rb') as f:
            pmod = pickle.load(f)

        Ey = pmod.predict(X)

        assert Ey.shape == y.shape
