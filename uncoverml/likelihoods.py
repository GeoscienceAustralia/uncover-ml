"""
Likelihood functions that can be used with revrand.

Can be used with revrand's GeneralisedLinearModel class for specialised
regression tasks such as basement depth estimation from censored and
uncensored depth observations.
"""

from __future__ import division

import numpy as np
from scipy.special import erf

from revrand.likelihoods import Bernoulli, Gaussian
from revrand.btypes import Parameter, Positive


# Constants
SQRTPI2 = np.sqrt(np.pi / 2)


class Switching(Bernoulli):

    def __init__(self, lenscale=1., var_init=Parameter(1., Positive())):

        self.params = var_init
        self.gaus = Gaussian(var_init)
        self.unif = UnifGauss(lenscale)

    def loglike(self, y, f, var, z):
        loglike = self.__split_on_z(self.unif.loglike, self.gaus.loglike, y, f,
                                    var, z)
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

    def dp(self, y, f, var, z):
        yz, fz = np.broadcast_arrays(y[z], f[:, z])
        dp = np.zeros_like(f)
        dp[:, z] = self.gaus.dp(yz, fz, var)
        return dp

    def __split_on_z(self, func_unif, func_gaus, y, f, var, z):

        y, f, z = np.broadcast_arrays(y, f, z)
        val = np.zeros_like(f)
        nz = ~ z
        val[z] = func_gaus(y[z], f[z], var)
        val[nz] = func_unif(y[nz], f[nz])

        return val


class UnifGauss(Bernoulli):

    def __init__(self, lenscale=1.):

        self.l = lenscale

    def loglike(self, y, f):

        ll = self.__split_apply(self._loglike_y_lt_f,
                                self._loglike_y_gt_f,
                                self._loglike_f_lt_0, y, f)

        return ll

    def _loglike_y_lt_f(self, y, f):

        return - np.log(f + self.l * SQRTPI2)

    def _loglike_y_gt_f(self, y, f):

        return self._loglike_y_lt_f(y, f) - 0.5 * ((y - f) / self.l)**2

    def _loglike_f_lt_0(self, y, f):

        return - np.log(self.l * SQRTPI2) - 0.5 * ((y - f) / self.l)**2

    def pdf(self, y, f):

        p = self.__split_apply(self._pdf_y_lt_f,
                               self._pdf_y_gt_f,
                               self._pdf_f_lt_0,
                               y, f)

        return p

    def _pdf_y_lt_f(self, y, f):

        return 1 / (f + self.l * SQRTPI2)

    def _pdf_y_gt_f(self, y, f):

        return np.exp(-0.5 * ((y - f) / self.l)**2) * self._pdf_y_lt_f(y, f)

    def _pdf_f_lt_0(self, y, f):

        return np.exp(-0.5 * ((y - f) / self.l)**2) / (self.l * SQRTPI2)

    def cdf(self, y, f):

        P = self.__split_apply(self._cdf_y_lt_f,
                               self._cdf_y_gt_f,
                               self._cdf_f_lt_0,
                               y, f)

        return P

    def _cdf_y_lt_f(self, y, f):

        return y / (f + self.l * SQRTPI2)

    def _cdf_y_gt_f(self, y, f):

        return (erf((y - f) / (self.l * np.sqrt(2))) * self.l * SQRTPI2 + f) \
            / (f + self.l * SQRTPI2)

    def _cdf_f_lt_0(self, y, f):

        return erf((y - f) / (self.l * np.sqrt(2)))

    def Ey(self, f):

        maxf0 = np.maximum(0, f)
        y_gt_f_mean = f + self.l / SQRTPI2
        y_lt_f_mean = maxf0 / 2

        norm = 1 / (maxf0 + self.l * SQRTPI2)
        weight_y_gt_f = norm * self.l * SQRTPI2
        weight_y_lt_f = maxf0 * norm

        Ey = weight_y_gt_f * y_gt_f_mean + weight_y_lt_f + y_lt_f_mean

        return Ey

    def df(self, y, f):

        df = self.__split_apply(self._df_y_lt_f,
                                self._df_y_gt_f,
                                self._df_f_lt_0,
                                y, f)

        return df

    def _df_y_lt_f(self, y, f):

        df = - 1. / (f + self.l * SQRTPI2)
        return df

    def _df_y_gt_f(self, y, f):

        df = self._df_y_lt_f(y, f) + self._df_f_lt_0(y, f)
        return df

    def _df_f_lt_0(self, y, f):

        df = (y - f) / self.l**2
        return df

    def __split_apply(self, func_y_lt_f, func_y_gt_f, func_f_lt_0, y, f):

        # Make sure all arrays are of a compatible type
        y, f = np.broadcast_arrays(y, f)
        y = y.astype(float, copy=False)
        f = f.astype(float, copy=False)

        if any(y < 0):
            raise ValueError("y has to be > 0!")

        # get indicators of which likelihoods to apply where
        y_lt_f = np.logical_and(y <= f, f > 0)
        y_gt_f = np.logical_and(y > f, f > 0)
        f_lt_0 = f <= 0

        result = np.zeros_like(y)

        result[y_lt_f] = func_y_lt_f(y[y_lt_f], f[y_lt_f])
        result[y_gt_f] = func_y_gt_f(y[y_gt_f], f[y_gt_f])
        result[f_lt_0] = func_f_lt_0(y[f_lt_0], f[f_lt_0])

        return result
