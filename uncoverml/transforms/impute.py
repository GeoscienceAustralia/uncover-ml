import logging

import numpy as np
from scipy.linalg import pinv, solve

from uncoverml import mpiops

log = logging.getLogger(__name__)


def impute_with_mean(x, mean):

    # No missing data
    if np.ma.count_masked(x) == 0:
        return x

    for i, m in enumerate(mean):
        x.data[:, i][x.mask[:, i]] = m
    x = np.ma.MaskedArray(data=x.data, mask=False)
    return x


class MeanImputer:
    def __init__(self):
        self.mean = None

    def __call__(self, x):
        if self.mean is None:
            self.mean = mpiops.mean(x)
        x = impute_with_mean(x, self.mean)
        return x


class GaussImputer:

    def __init__(self):
        self.mean = None
        self.prec = None

    def __call__(self, x):

        if self.mean is None or self.prec is None:
            self._make_impute_stats(x)

        for i in range(len(x)):
            x[i].mask = False
            self._gaus_condition(x[i])

    def _make_impute_stats(self, x):

        self.mean = mpiops.mean(x)
        cov = mpiops.covariance(x)
        self.prec = pinv(cov)  # SVD pseudo inverse

    def _gaus_condition(self, xi):

        if np.ma.count_masked(xi) == 0:
            return

        a = xi.mask
        b = ~xi.mask

        xb = xi[b].data
        ma = self.mean[a]
        mb = self.mean[b]
        Laa = self.prec[np.ix_(a, a)]
        Lab = self.prec[np.ix_(a, b)]

        xi[a] = ma - solve(Laa, Lab.dot(xb - mb), sym_pos=True)
