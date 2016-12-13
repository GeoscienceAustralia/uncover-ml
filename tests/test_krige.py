from itertools import product

import numpy as np
import pytest
from pykrige.ok import OrdinaryKriging

from uncoverml.krige import Krige, krige_methods

data = np.array([[0.0, 0, 0.47],
                 [1.9, 0.6, 0.56],
                 [1.1, 3.2, 0.74],
                 [0, 2.5, 1.47],
                 [4.75, 3.8, 1.74]])

gridx = np.arange(0.0, 5.5, 0.5)
gridy = np.arange(0.0, 5.5, 0.5)

OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                     variogram_model='linear',
                     verbose=False, enable_plotting=False)

z, ss = OK.execute('grid', gridx, gridy)


def test_supplied():
    assert round(z[0, 0]-0.47, 6) == 0
    assert round(z[5, 0] - 1.47, 6) == 0


def test_grid_vs_points():
    points = list(product(gridx, gridy))
    points_x = [p[0] for p in points]
    points_y = [p[1] for p in points]
    zp, ssp = OK.execute('points', points_y, points_x)
    zpp = zp.reshape(z.shape)
    assert np.allclose(z, zpp)


@pytest.fixture(params=['ordinary', 'universal'])
def krig_method(request):
    return request.param


def test_krige(krig_method):
    k = Krige(method=krig_method, n_closest_points=2)
    k.fit(x=data[:, :2], y=data[:, 2])
    points = np.array(list(product(gridx, gridy)))
    points_x = [p[0] for p in points]
    points_y = [p[1] for p in points]
    OUK = krige_methods[krig_method](data[:, 0], data[:, 1], data[:, 2],
                                     variogram_model='linear',
                                     verbose=False, enable_plotting=False)
    if isinstance(OUK, OrdinaryKriging):
        zp, ssp = OUK.execute('points', points_x, points_y,
                              n_closest_points=2,
                              backend='loop')
    else:
        zp, ssp = OUK.execute('points', points_x, points_y)
    assert round(zp[0] - 0.47, 6) == 0
    assert round(zp[5] - 1.47, 6) == 0
    assert np.allclose(k.predict(points), zp)
