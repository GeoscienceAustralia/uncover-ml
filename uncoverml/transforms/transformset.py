import copy
import logging

import numpy as np

from uncoverml import mpiops

log = logging.getLogger(__name__)


def build_feature_vector(image_chunks):
    for k, im in image_chunks.items():
        image_chunks[k] = im.reshape(im.shape[0], -1)
    x = np.ma.concatenate(image_chunks.values(), axis=1)
    return x


def missing_percentage(x):
    x_n = np.sum(mpiops.count(x))
    x_full_local = np.product(x.shape)
    x_full = mpiops.comm.allreduce(x_full_local)
    missing = (1.0 - x_n / x_full) * 100.0
    return missing


class TransformSet:
    def __init__(self, imputer=None, transforms=None):
            self.global_transforms = (transforms if transforms else [])
            self.imputer = imputer

    def __call__(self, x):
        # impute
        if self.imputer:
            missing_percent = missing_percentage(x)
            log.info("Imputing {:2.2f}% missing data".format(missing_percent))
            x = self.imputer(x)

        # transforms
        for t in self.global_transforms:
            x = t(x)

        return x


class ImageTransformSet(TransformSet):
    def __init__(self, image_transforms=None, imputer=None,
                 global_transforms=None):
            self.image_transforms = (image_transforms if image_transforms
                                     else [])
            super().__init__(imputer, global_transforms)

    def __call__(self, image_chunks):
        transformed_chunks = copy.copy(image_chunks)
        # apply the per-image transforms
        for lbl in image_chunks:
            for t in self.image_transforms:
                transformed_chunks[lbl] = t(transformed_chunks[lbl])

        # concatenate and floating point
        x = build_feature_vector(transformed_chunks)

        x = super().__call__(x)
        return x
