import logging

import numpy as np

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
