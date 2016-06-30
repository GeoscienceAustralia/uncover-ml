from __future__ import division

import numpy as np
from scipy.special import digamma, betaln
from scipy.stats import beta

from revrand.likelihoods import Bernoulli
from revrand.btypes import Parameter, Positive


class Beta3(Bernoulli):
    """
    A three-parameter Beta distribution,

    .. math::

        \mathcal{B}(y | f, \alpha, \beta) = \frac{1}{f^{\alpha + \beta - 1}
            B(\alpha, \beta)} y^{\alpha - 1} (f - y)^{\beta - 1},

    where :math:`B(\cdot)` is a Beta function. This is a distribution between
    :math:`(0, f)`, with the special case of :math:`\alpha = \beta = 1` being a
    uniform distribution.

    Parameters
    ----------
    a_init: Parameter, optional
        A scalar Parameter describing the initial point and bounds for
        an optimiser to learn the a-shape parameter of this object.
    b_init: Parameter, optional
        A scalar Parameter describing the initial point and bounds for
        an optimiser to learn the b-shape parameter of this object.
    """

    def __init__(self,
                 a_init=Parameter(1., Positive()),
                 b_init=Parameter(1., Positive())
                 ):

        self.params = [a_init, b_init]

    def loglike(self, y, f, a, b):
        r"""
        Three-parameter Beta log likelihood.

        Parameters
        ----------
        y: ndarray
            array of (0, f) valued targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        a: float
            shape parameter, a > 0
        b: float
            shape parameter, b > 0

        Returns
        -------
        logp: ndarray
            the log likelihood of each y given each f under this
            likelihood.
        """

        self.__check_ab(a, b)

        yok, ybad, fok, fbad, okind, badind = self.__y_gte_f(y, f)
        log_like = np.zeros_like(f)

        norm_const = -(a + b - 1) * np.log(fok) - betaln(a, b)
        like = (a - 1) * np.log(yok) + (b - 1) * np.log(fok - yok)
        log_like[okind] = norm_const + like

        if len(badind) > 0:
            penalty = - np.exp(np.abs(ybad - fbad))
            log_like[badind] = penalty

        return log_like

    def Ey(self, f, a, b):
        r""" Expected value of the three-parameter Beta.

        Parameters
        ----------
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        a: float
            shape parameter, a > 0
        b: float
            shape parameter, b > 0

        Returns
        -------
        Ey: ndarray
            expected value of y, :math:`\mathbb{E}[y|f]`.
        """

        self.__check_ab(a, b)

        return (a * f) / (a + b)

    def df(self, y, f, a, b):
        r"""
        Derivative of three-parameter Beta log likelihood w.r.t.\  f.

        Parameters
        ----------
        y: ndarray
            array of (0, f) valued targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        a: float
            shape parameter, a > 0
        b: float
            shape parameter, b > 0

        Returns
        -------
        df: ndarray
            the derivative :math:`\partial \log p(y|f) / \partial f`
        """

        self.__check_ab(a, b)
        yok, ybad, fok, fbad, okind, badind = self.__y_gte_f(y, f)
        df = np.zeros_like(f)

        df[okind] = (b - 1) / (fok - yok) - (a + b + 1) / fok

        if len(badind) > 0:
            df[badind] = np.exp(np.abs(ybad - fbad))

        return df

    def d2f(self, y, f, a, b):
        r"""
        Second derivative of three-parameter Beta log likelihood w.r.t.\  f.

        Parameters
        ----------
        y: ndarray
            array of (0, f) valued targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        a: float
            shape parameter, a > 0
        b: float
            shape parameter, b > 0

        Returns
        -------
        df: ndarray
            the second derivative
            :math:`\partial^2 \log p(y|f)/ \partial f^2`
        """

        self.__check_ab(a, b)
        yok, ybad, fok, fbad, okind, badind = self.__y_gte_f(y, f)
        d2f = np.zeros_like(f)

        d2f[okind] = (a + b + 1) / fok**2 - (b - 1) / (fok - yok)**2

        if len(badind) > 0:
            d2f[badind] = - np.exp(np.abs(ybad - fbad))

        return d2f

    def d3f(self, y, f, a, b):
        r"""
        Third derivative of three-parameter Beta log likelihood w.r.t.\  f.

        Parameters
        ----------
        y: ndarray
            array of (0, f) valued targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        a: float
            shape parameter, a > 0
        b: float
            shape parameter, b > 0

        Returns
        -------
        df: ndarray
            the third derivative
            :math:`\partial^3 \log p(y|f)/ \partial f^3`
        """

        self.__check_ab(a, b)
        yok, ybad, fok, fbad, okind, badind = self.__y_gte_f(y, f)
        d3f = np.zeros_like(f)

        d3f[okind] = 2 * (b - 1) / (fok - yok)**3 - 2 * (a + b + 1) / fok**3

        if len(badind) > 0:
            d3f[badind] = np.exp(np.abs(ybad - fbad))

        return d3f

    def dp(self, y, f, a, b):
        r"""
        Derivatives of three-parameter Beta log likelihood w.r.t.\ the
        parameters, :math:`a` and math:`b`.

        Parameters
        ----------
        y: ndarray
            array of (0, f) valued targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        a: float
            shape parameter, a > 0
        b: float
            shape parameter, b > 0

        Returns
        -------
        dp: list
            the derivatives
            :math:`\partial \log p(y|f, a, b)/ \partial a` and
            :math:`\partial \log p(y|f, a, b)/ \partial b` and
        """

        self.__check_ab(a, b)
        yok, _, fok, _, _, _ = self.__y_gte_f(y, f)

        digamma_ab = digamma(a + b)

        da = digamma_ab - digamma(a) - np.log(fok) + np.log(yok)
        db = digamma_ab - digamma(b) + np.log(1 - yok / fok)

        return [da, db]

    def dpd2f(self, y, f, a, b):
        r"""
        Partial derivative of three-parameter Beta log likelihood,
        :math:`\partial h(f, a, b) / \partial a` and
        :math:`\partial h(f, a, b) / \partial b` and where
        :math:`h(f, a, b) = \partial^2 \log p(y|f, a, b)/ \partial f^2`.

        Parameters
        ----------
        y: ndarray
            array of (0, f) valued targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        a: float
            shape parameter, a > 0
        b: float
            shape parameter, b > 0

        Returns
        -------
        dpd2f: list
            the derivative of the likelihood Hessian w.r.t.\ the parameters
            :math:`a` and :math:`b`.
        """

        self.__check_ab(a, b)
        yok, _, fok, _, _, _ = self.__y_gte_f(y, f)

        da = 1. / fok**2
        db = da - 1. / (fok - yok)**2
        return [da, db]

    def cdf(self, y, f, a, b):
        r"""
        Cumulative density function of the likelihood.

        Parameters
        ----------
        y: ndarray
            array of (0, f) valued targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        a: float
            shape parameter, a > 0
        b: float
            shape parameter, b > 0

        Returns
        -------
        cdf: ndarray
            Cumulative density function evaluated at y.
        """

        self.__check_ab(a, b)

        return beta.cdf(y / f, a, b)

    def pdf(self, y, f, a, b):

        self.__check_ab(a, b)

        return beta.pdf(y / f, a, b) / f

    def rvs(self, f, a, b):

        self.__check_ab(a, b)

        return beta.rvs(a, b, size=f.shape) * f

    def __check_ab(self, a, b):

        if a <= 0:
            raise ValueError("a must be greater than 0")

        if b <= 0:
            raise ValueError("b must be greater than 0")

    def __y_gte_f(self, y, f):

        if np.isscalar(y) and np.isscalar(f):
            y = np.array([y])
            f = np.array([f])
        elif np.isscalar(y) and not np.isscalar(f):
            y = y * np.ones_like(f)
        elif np.isscalar(f) and not np.isscalar(y):
            f = f * np.ones_like(y)

        bad_ind = y >= f
        ok_ind = ~bad_ind
        yok, fok = y[ok_ind], f[ok_ind]
        ybad, fbad = y[bad_ind], f[bad_ind]

        return yok, ybad, fok, fbad, np.where(ok_ind)[0], np.where(bad_ind)[0]


class AsymmetricLaplace(Bernoulli):

    def __init__(self, scale_init=Parameter(1., Positive()), asymmetry=1.):

        self.params = scale_init
        self.kappa = asymmetry

    def pdf(self, y, f, scale):

        return np.exp(self.loglike(y, f, scale))

    def rvs(self, f, scale):

        U = np.random.rand(*f.shape) \
            * (1. / self.kappa + self.kappa) - self.kappa

        s = np.sign(U)
        skappas = self.__skappas(s)
        norm = 1. / (scale * skappas)

        return f - norm * np.log(U * skappas)

    def loglike(self, y, f, scale):

        norm = np.log(scale) - np.log(self.kappa + 1. / self.kappa)
        yf, s = self.__yfs(y, f)

        return norm - yf * scale * self.__skappas(s)

    def Ey(self, f, scale):

        return f + (1 - self.kappa**2) / (scale * self.kappa)

    def df(self, y, f, scale):

        yf, s = self.__yfs(y, f)

        return scale * self.__skappas(s)

    def d2f(self, y, f, scale):

        return np.zeros_like(f)

    def d3f(self, y, f, scale):

        return np.zeros_like(f)

    def dp(self, y, f, scale):

        yf, s = self.__yfs(y, f)

        return 1. / scale - yf * self.__skappas(s)

    def dpd2f(self, y, f, scale):

        return np.zeros_like(f)

    def cdf(self, y, f, scale):

        yf, s = self.__yfs(y, f)
        kappa2 = self.kappa**2

        cdf = np.exp(-yf * scale * self.__skappas(s)) / (1 + kappa2)
        cdf[y <= f] = cdf[y <= f] * kappa2
        cdf[y > f] = 1 - cdf[y > f]

        return cdf

    def __yfs(self, y, f):

        yf = y - f
        s = np.sign(yf)

        return yf, s

    def __skappas(self, s):

        return s * np.power(self.kappa, s)
