import numpy as np
import pytest
from revrand.btypes import Parameter, Positive
from uncoverml.likelihoods import Switching, UnifGauss  # adjust `your_module` accordingly


@pytest.fixture
def dummy_inputs():
    y = np.array([0.5, 1.0, 1.5])
    f = np.array([0.3, 1.2, 1.4])
    z = np.array([True, False, True])
    var = 1.0
    return y, f, z, var


def test_switching_loglike(dummy_inputs):
    y, f, z, var = dummy_inputs
    model = Switching()
    loglike = model.loglike(y, f, var, z)
    assert loglike.shape == f.shape
    assert np.all(np.isfinite(loglike))


def test_switching_ey(dummy_inputs):
    _, f, z, var = dummy_inputs
    model = Switching()
    ey = model.Ey(f, var, z)
    assert ey.shape == f.shape
    assert np.all(np.isfinite(ey))


def test_unifgauss_loglike_positive():
    model = UnifGauss(lenscale=1.0)
    y = np.array([0.5, 1.0, 2.0])
    f = np.array([1.0, 1.5, -1.0])
    loglike = model.loglike(y, f)
    assert loglike.shape == y.shape
    assert np.all(np.isfinite(loglike))


def test_unifgauss_loglike_raises():
    model = UnifGauss()
    y = np.array([-0.1, 0.5])
    f = np.array([0.0, 0.5])
    with pytest.raises(ValueError):
        model.loglike(y, f)
