import numpy as np

from scipy.special import expit, logit
from scipy.stats import gaussian_kde, norm
from scipy.optimize import brentq
from scipy.special import erfinv


class Identity():

    def fit(self, y):
        pass

    def transform(self, y):
        return y

    def itransform(self, y_transformed):
        return y_transformed


class Standardise(Identity):

    def fit(self, y):

        self.ymean = y.mean()
        self.ystd = y.std()

    def transform(self, y):

        y_transformed = (y - self.ymean) / self.ystd
        return y_transformed

    def itransform(self, y_transformed):

        y_t = y_transformed * self.ystd + self.ymean
        return self.transformer.itransform(y_t)


class Sqrt(Identity):

    def transform(self, y):

        ysqrt = np.sqrt(y)
        return ysqrt

    def itransform(self, y_transformed):

        return y_transformed**2


class Log(Identity):

    def __init__(self, offset=0.):

        self.offset = offset

    def transform(self, y):

        ylog = np.log(y + self.offset)
        return ylog

    def itransform(self, y_transformed):

        return np.exp(y_transformed) - self.offset


class Logistic(Identity):

    def __init__(self, scale):

        self.scale = scale

    def transform(self, y):

        yexpit = expit(self.scale * y)

        return yexpit

    def itransform(self, y_transformed):

        yscale = logit(y_transformed)
        return (yscale / self.scale)


class RankGaussian(Identity):
    """Forces the marginal histogram to be Gaussian."""

    def fit(self, y):
        x = y[~np.isnan(y)]
        n = len(y)
        self.y = erfinv(np.linspace(0., 1., n + 2)[1:-1])
        self.s = np.sort(x)

    def transform(self, y):
        return np.interp(y, self.s, self.y)

    def itransform(self, y):
        return np.interp(y, self.y, self.s)


class KDE(Identity):

    def fit(self, y):

        self.lb = y.min() - 1e6
        self.ub = y.max() + 1e6
        self.kde = gaussian_kde(y)

    def transform(self, y):

        ycdf = [self.kde.integrate_box_1d(-np.inf, yi) for yi in y]
        ygauss = norm.isf(1 - np.array(ycdf))
        return ygauss

    def itransform(self, y_transformed):

        ycdf = norm.cdf(y_transformed)
        y = [brentq(self._obj, a=self.lb, b=self.ub, args=(yi,))
             for yi in ycdf]
        return np.array(y)

    def _obj(self, q, percent):

        return self.kde.integrate_box_1d(-np.inf, q) - percent


transforms = {'indentity': Identity,
              'standardise': Standardise,
              'sqrt': Sqrt,
              'log': Log,
              'logistic': Logistic,
              'rank': RankGaussian,
              'kde': KDE
              }
