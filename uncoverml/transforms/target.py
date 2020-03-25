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
        self.ystd = (y - self.ymean).std()

    def transform(self, y):
        y_transformed = (y - self.ymean) / self.ystd
        return y_transformed

    def itransform(self, y_transformed):
        y = (y_transformed * self.ystd) + self.ymean
        return y


class Sqrt(Identity):

    def __init__(self, offset=0.):

        self.offset = offset

    def transform(self, y):

        ysqrt = np.sqrt(y + self.offset)
        return ysqrt

    def itransform(self, y_transformed):

        return y_transformed**2 - self.offset


class Log(Identity):

    def __init__(self, offset=0., replace_zeros=True):

        self.offset = offset
        self.replace_zeros = replace_zeros

    def fit(self, y):

        self.ymin = y[(y + self.offset) > 0].min()

    def transform(self, y):

        y = y + self.offset
        if self.replace_zeros:
            if isinstance(y, np.ma.masked_array):
                y._sharedmask = False
            y[y == 0] = self.ymin / 10.

        return np.log(y)

    def itransform(self, y_transformed):

        return np.exp(y_transformed) - self.offset


class Logistic(Identity):

    def __init__(self, scale=1):

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

    def itransform(self, y_transformed):
        return np.interp(y_transformed, self.y, self.s)


class KDE(Identity):

    def fit(self, y):

        self.lb = y.min() - 1e6
        self.ub = y.max() + 1e6
        self.kde = gaussian_kde(y)

    def transform(self, y):

        # FIXME: Interpolate rather than solve for speed?
        ycdf = [self.kde.integrate_box_1d(-np.inf, yi) for yi in y]
        ygauss = norm.isf(1 - np.array(ycdf))
        return ygauss

    def itransform(self, y_transformed):

        # FIXME: Interpolate rather than solve for speed?
        ycdf = norm.cdf(y_transformed)
        y = [brentq(self._obj, a=self.lb, b=self.ub, args=(yi,))
             for yi in ycdf]
        return np.array(y)

    def _obj(self, q, percent):

        return self.kde.integrate_box_1d(-np.inf, q) - percent


transforms = {'identity': Identity,
              'standardise': Standardise,
              'sqrt': Sqrt,
              'log': Log,
              'logistic': Logistic,
              'rank': RankGaussian,
              'kde': KDE
              }
