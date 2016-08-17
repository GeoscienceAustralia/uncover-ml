import logging

import numpy as np
from uncoverml import validation

log = logging.getLogger(__name__)


class ImageDataVector:

    def __init__(self, x, origin, pix_size, patchsize):
        self.x = x
        self.origin = origin
        self.pix_size = pix_size
        self.patchsize = patchsize
        pass


class Settings:

    def __repr__(self):
        return str(self.__dict__)


class ExtractSettings(Settings):

    def __init__(self, onehot, x_sets, patchsize):
        self.onehot = onehot
        self.x_sets = x_sets
        self.patchsize = patchsize


class ComposeSettings(Settings):

    def __init__(self, impute, transform, featurefraction, impute_mean, mean,
                 sd, eigvals, eigvecs):
        self.impute = impute
        self.transform = transform
        self.featurefraction = featurefraction
        self.impute_mean = impute_mean
        self.mean = mean
        self.sd = sd
        self.eigvals = eigvals
        self.eigvecs = eigvecs


class CrossValTargets:

    def __init__(self, lonlat, vals, folds=10, seed=None,
                 sort=False, othervals=None):
        self.nfolds = folds
        N = len(lonlat)
        self.fields = {}

        # we may be given folds already
        if type(folds) == int:
            _, cvassigns = validation.split_cfold(N, folds, seed)
        else:
            cvassigns = folds
        if sort:
            # Get ascending order of targets by lat then lon
            # FIXME -- temporary hack, only works with y_pix_size < 0
            # ordind = np.lexsort(lonlat.T)[::-1]
            ordind = np.lexsort(lonlat.T)
            self.observations = vals[ordind]
            self.positions = lonlat[ordind]
            self.folds = cvassigns[ordind]
            self._observations_unsorted = vals
            self._positions_unsorted = lonlat
            self._folds_unsorted = cvassigns
            if othervals is not None:
                self.fields = {k: v[ordind] for k, v in othervals.items()}
                self._fields_unsorted = othervals
        else:
            self.observations = vals
            self.positions = lonlat
            self.folds = cvassigns
            if othervals is not None:
                self.fields = othervals
