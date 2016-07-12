from __future__ import division

import numpy as np
from scipy.special import erf, expit
from scipy.stats import norm

from revrand.likelihoods import Bernoulli, Gaussian
from revrand.btypes import Parameter, Positive
from revrand.mathfun.special import softplus


# Constants
SQRTPI2 = np.sqrt(np.pi) / 2


class Switching(Bernoulli):

    def __init__(self, var_init=Parameter(1., Positive())):

        self.params = var_init
        self.gaus = Gaussian(var_init)
        self.unif = UnifGauss()

        # FIXME: TEMP
        self.i = 0

    def loglike(self, y, f, var, z):
        loglike = self.__split_on_z(self.unif.loglike, self.gaus.loglike, y, f,
                                    var, z)
        # FIXME: TEMP
        if (self.i % 1000) == 0:
            print("fmin: {}, fmax: {}, fshape: {}".format(f.min(), f.max(),
                                                          f.shape))
        self.i += 1
        return loglike

    def Ey(self, f, var, z):
        f, z = np.broadcast_arrays(f, z)
        Ey = np.zeros_like(f)
        nz = ~ z

        Ey[z] = self.gaus.Ey(f[z], var)
        Ey[nz] = self.unif.Ey(f[nz])
        return Ey

    def cdf(self, y, f, var, z):
        cdf = self.__split_on_z(self.unif.cdf, self.gaus.cdf, y, f, var, z)
        return cdf

    def df(self, y, f, var, z):
        df = self.__split_on_z(self.unif.df, self.gaus.df, y, f, var, z)
        return df

    def d2f(self, y, f, var, z):
        d2f = self.__split_on_z(self.unif.d2f, self.gaus.d2f, y, f, var, z)
        return d2f

    def d3f(self, y, f, var, z):
        d3f = self.__split_on_z(self.unif.d3f, self.gaus.d3f, y, f, var, z)
        return d3f

    def dp(self, y, f, var, z):
        dp = np.zeros_like(f)
        dp[z] = self.gaus.dp(y[z], f[z], var)
        return dp

    def dpd2f(self, y, f, var, z):
        dpd2f = np.zeros_like(f)
        dpd2f[z] = self.gaus.dpd2f(y[z], f[z], var)
        return dpd2f

    def __split_on_z(self, func_unif, func_gaus, y, f, var, z):

        y, f, z = np.broadcast_arrays(y, f, z)
        val = np.zeros_like(f)
        nz = ~ z
        val[z] = func_gaus(y[z], f[z], var)
        val[nz] = func_unif(y[nz], f[nz])

        return val


class PosGaussian(Gaussian):

    def loglike(self, y, f, var):

        return norm.logpdf(y, loc=softplus(f), scale=np.sqrt(var))

    def Ey(self, y, f, var):

        return softplus(f)

    def df(self, y, f, var):

        g = softplus(f)
        gp = expit(f)

        df = (y - g) * gp / var
        return df

    def d2f(self, y, f, var):

        g = softplus(f)
        gp = expit(f)
        g2p = gp * (1 - gp)

        d2f = ((y - g) * g2p - gp**2) / var
        return d2f

    def d3f(self, y, f, var):

        g = softplus(f)
        gp = expit(f)
        g2p = gp * (1 - gp)
        g3p = g2p * (1 - 2 * gp)

        d3f = ((y - g) * g3p - 3 * gp * g2p) / var
        return d3f

    def dp(self, y, f, var):

        g = softplus(f)
        ivar = 1. / var
        return 0.5 * (((y - g) * ivar)**2 - ivar)

    def dpd2f(self, y, f, var):

        dpd2f = - self.d2f(y, f, var) / var
        return dpd2f

    def cdf(self, y, f, var):

        return norm.cdf(y, loc=softplus(f), scale=np.sqrt(var))


class UnifGauss(Bernoulli):

    def loglike(self, y, f):

        return self.__split_apply(self._loglike_unif, self._loglike_gaus, y, f)

    def _loglike_unif(self, y, f, g):

        return - np.log(g + SQRTPI2)

    def _loglike_gaus(self, y, f, g):

        return - np.log(g + SQRTPI2) - (y - g)**2

    def pdf(self, y, f):

        return self.__split_apply(self._pdf_unif, self._pdf_gaus, y, f)

    def _pdf_unif(self, y, f, g):

        return 1 / (g + SQRTPI2)

    def _pdf_gaus(self, y, f, g):

        return np.exp(-(y - g)**2) / (g + SQRTPI2)

    def cdf(self, y, f):

        return self.__split_apply(self._cdf_unif, self._cdf_gaus, y, f)

    def _cdf_unif(self, y, f, g):

        return y / (g + SQRTPI2)

    def _cdf_gaus(self, y, f, g):

        return (erf(y - g) * SQRTPI2 + g) / (g + SQRTPI2)

    def Ey(self, f):

        g = softplus(f)

        gaus_mean = g + np.sqrt(2 / np.pi)
        unif_mean = g / 2

        norm = 1 / (g + SQRTPI2)
        weight_gaus = norm * SQRTPI2
        weight_unif = g * norm

        Ey = weight_gaus * gaus_mean + weight_unif + unif_mean

        return Ey

    def df(self, y, f):

        return self.__split_apply(self._df_unif, self._df_gaus, y, f)

    def _df_unif(self, y, f, g, gdivs=False):

        gp = expit(f)
        df = - gp / (SQRTPI2 + g)

        return df if not gdivs else (df, gp)

    def _df_gaus(self, y, f, g):

        gp = expit(f)
        dnorm, gp = self._df_unif(y, f, g, gdivs=True)
        return dnorm + 2 * gp * (y - g)

    def d2f(self, y, f):

        return self.__split_apply(self._d2f_unif, self._d2f_gaus, y, f)

    def _d2f_unif(self, y, f, g, gdivs=False):

        gp = expit(f)
        g2p = gp * (1 - gp)
        SQRTPI2g = (SQRTPI2 + g)
        d2f = (gp / SQRTPI2g)**2 - g2p / SQRTPI2g

        return d2f if not gdivs else (d2f, gp, g2p)

    def _d2f_gaus(self, y, f, g):

        d2norm, gp, g2p = self._d2f_unif(y, f, g, gdivs=True)
        return d2norm + 2 * (g2p * (y - g) - gp**2)

    def d3f(self, y, f):

        return self.__split_apply(self._d3f_unif, self._d3f_gaus, y, f)

    def _d3f_unif(self, y, f, g, gdivs=False):

        gp = expit(f)
        g2p = gp * (1 - gp)
        g3p = g2p * (1 - 2 * gp)
        SQRTPI2g = (SQRTPI2 + g)
        d3f = - g3p / SQRTPI2g + 3 * gp * g2p / SQRTPI2g**2 \
            - 2 * (gp / SQRTPI2g)**3

        return d3f if not gdivs else (d3f, gp, g2p, g3p)

    def _d3f_gaus(self, y, f, g):

        d3norm, gp, g2p, g3p = self._d3f_unif(y, f, g, gdivs=True)
        d3f_gaus = d3norm + 2 * g3p * (y - g) - 6 * g2p * gp
        return d3f_gaus

    def __split_apply(self, func_unif, func_gaus, y, f):

        y, f = np.broadcast_arrays(y, f)
        g = softplus(f)
        isunif = y <= g
        isgaus = ~ isunif

        result = np.zeros_like(y)

        result[isunif] = func_unif(y[isunif], f[isunif], g[isunif])
        result[isgaus] = func_gaus(y[isgaus], f[isgaus], g[isgaus])

        return result
