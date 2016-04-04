import pickle
from os import path

from uncoverml import models
from uncoverml import validation


def test_modelmap(make_fakedata):

    X, y, _, _ = make_fakedata

    for model in models.modelmaps.keys():
        mod = models.modelmaps[model]()
        mod.fit(X, y)

        if model != 'bayesreg':
            Ey = mod.predict(X)
        else:
            Ey, _, _ = mod.predict(X)

        assert validation.rsquare(Ey, y) > 0


def test_modelpersistance(make_fakedata):

    X, y, _, mod_dir = make_fakedata

    for model in models.modelmaps.keys():
        mod = models.modelmaps[model]()
        mod.fit(X, y)

        with open(path.join(mod_dir, model + ".pk"), 'wb') as f:
            pickle.dump(mod, f)

        with open(path.join(mod_dir, model + ".pk"), 'rb') as f:
            pmod = pickle.load(f)

        if model != 'bayesreg':
            Ey = pmod.predict(X)
        else:
            Ey, _, _ = pmod.predict(X)

        assert Ey.shape == y.shape
