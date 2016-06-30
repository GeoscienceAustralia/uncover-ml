import numpy as np
from scipy.stats import laplace, beta

from uncoverml.likelihoods import AsymmetricLaplace, Beta3


def test_asym_laplace():

    # Test we can at least match a Laplace distribution

    kappa = 1
    alap = AsymmetricLaplace(asymmetry=kappa)

    scale = 2
    x = np.linspace(-10, 10, 100)

    p1 = laplace.logpdf(x, scale=2)
    p2 = alap.loglike(x, 0, scale)

    np.allclose(p1, p2)


def test_beta3():

    # Test we can at least match a Laplace distribution

    a = 2
    b = 5

    beta3 = Beta3()

    x = np.linspace(0 + 1e-3, 1 - 1e-3, 100)

    p1 = beta.logpdf(x, a, b)
    p2 = beta3.loglike(x, 1, a, b)

    np.allclose(p1, p2)
