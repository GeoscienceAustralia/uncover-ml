import logging

import numpy as np
from scipy.linalg import pinv, solve
from scipy.spatial import cKDTree

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
    """
    Simple mean imputation.

    Replaces the missing values in x, with the mean of x.

    """

    def __init__(self):
        self.mean = None

    def __call__(self, x):
        if self.mean is None:
            self.mean = mpiops.mean(x)
        x = impute_with_mean(x, self.mean)
        return x


class GaussImputer:
    """
    Gaussian Imputer.

    This imputer fits a Gaussian to the data, then conditions on this Gaussian
    to interpolate missing data. This is effectively the same as using a linear
    regressor to impute the missing data, given all of the non-missing
    dimensions.

    Have a look at:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions

    We use the precision (inverse covariance) form of the Gaussian for
    computational efficiency.

    """

    def __init__(self):
        self.mean = None
        self.prec = None

    def __call__(self, x):

        if self.mean is None or self.prec is None:
            self._make_impute_stats(x)

        for i in range(len(x)):
            x.data[i] = self._gaus_condition(x[i])

        return np.ma.MaskedArray(data=x.data, mask=False)

    def _make_impute_stats(self, x):

        self.mean = mpiops.mean(x)
        cov = mpiops.covariance(x)
        self.prec, rank = pinv(cov, return_rank=True)  # stable pseudo inverse

        # if rank < len(self.mean):
        #     raise RuntimeError("This imputation method does not work on low "
        #                        "rank problems!")

    def _gaus_condition(self, xi):

        if np.ma.count_masked(xi) == 0:
            return xi

        a = xi.mask
        b = ~xi.mask

        xb = xi[b].data
        Laa = self.prec[np.ix_(a, a)]
        Lab = self.prec[np.ix_(a, b)]

        xfill = np.empty_like(xi)
        xfill[b] = xb
        xfill[a] = self.mean[a] - solve(Laa, Lab.dot(xb - self.mean[b]))
        return xfill


class NearestNeighboursImputer:
    """
    Nearest neighbour imputation.

    This builds up a KD tree using random points (without missing data), then
    fills in the missing data in query points with values from thier average
    nearest neighbours.

    Parameters
    ----------
    nodes: int, optional
        maximum number of points to use as nearest neightbours.
    k: int, optional
        number of neighbours to average for missing values.
    """

    def __init__(self, nodes=500, k=3):
        self.k = k
        self.nodes = nodes
        self.kdtree = None

    def __call__(self, x):

        # impute with neighbours
        missing_ind = np.ma.count_masked(x, axis=1) > 0

        if self.kdtree is None:
            self._make_kdtree(x)

        if missing_ind.sum() > 0:
            missing_mask = x.mask[missing_ind]
            nn = self._av_neigbours(x[missing_ind])
            x.data[x.mask] = nn[missing_mask]
        return np.ma.MaskedArray(data=x.data, mask=False)

    def _make_kdtree(self, x):
        self.kdtree = cKDTree(mpiops.random_full_points(x, Napprox=self.nodes))
        if not np.isfinite(self.kdtree.query(x, k=self.k)[0]).all():
            log.warning('Kdtree computation encountered problem. '
                        'Not enough neighbors available to compute '
                        'kdtree. Printing kdtree for debugging purpose')
            raise ValueError('Computed kdtree is not fully populated.'
                             'Not enough valid neighbours available.')

    def _get_neighbour(self, xq):
        _, neighbourind = self.kdtree.query(xq)
        return self.kdtree.data[neighbourind]

    def _av_neigbours(self, xq):

        xnn = [self.kdtree.data[self.kdtree.query(x, k=self.k)[1]].mean(axis=0)
               for x in xq]
        return np.vstack(xnn)
