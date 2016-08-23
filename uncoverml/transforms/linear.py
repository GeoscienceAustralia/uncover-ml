
import numpy as np

from uncoverml import mpiops


class CentreTransform:
    def __init__(self):
        self.mean = None

    def __call__(self, x):
        if self.mean is None:
            self.mean = mpiops.mean(x)
        x -= self.mean
        return x


class StandardiseTransform:
    def __init__(self):
        self.sd = None

    def __call__(self, x):

        if self.sd is None:
            self.sd = mpiops.sd(x)

        # remove dimensions with no st. dev. (and hence no info)
        zero_mask = self.sd == 0.
        if zero_mask.sum() > 0:
            x = x[:, ~zero_mask]
            sd = self.sd[~zero_mask]
        else:
            sd = self.sd
        x /= sd
        return x


class WhitenTransform:
    def __init__(self, keep_fraction):
        self.mean = None
        self.eigvals = None
        self.eigvecs = None
        self.keep_fraction = keep_fraction

    def __call__(self, x):

        if self.mean is None or self.eigvals is None or self.eigvecs is None:
            self.mean = mpiops.mean(x)
            self.eigvals, self.eigvecs = mpiops.eigen_decomposition(x)

        ndims = x.shape[1]
        # make sure 1 <= keepdims <= ndims
        keepdims = min(max(1, int(ndims * self.keep_fraction)), ndims)
        mat = self.eigvecs[:, -keepdims:]
        vec = self.eigvals[np.newaxis, -keepdims:]
        x = np.ma.dot(x - self.mean, mat, strict=True) / np.sqrt(vec)

        return x
