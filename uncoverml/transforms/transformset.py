import copy
import logging

import numpy as np

from uncoverml import mpiops

log = logging.getLogger(__name__)


def build_feature_vector(image_chunks, is_categorical):
    dtype = int if is_categorical else float
    for k, im in image_chunks.items():
        image_chunks[k] = im.reshape(im.shape[0], -1).astype(dtype)
    x_data = np.concatenate([a.data for a in image_chunks.values()], axis=1)
    x_mask = np.concatenate([a.mask for a in image_chunks.values()], axis=1)
    x = np.ma.masked_array(data=x_data, mask=x_mask)
    return x


def missing_percentage(x):
    x_n = np.sum(mpiops.count(x))
    x_full_local = np.product(x.shape)
    x_full = mpiops.comm_world.allreduce(x_full_local)
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
                 global_transforms=None, is_categorical=False):
        self.image_transforms = (image_transforms if image_transforms
                                 else [])
        self.is_categorical = is_categorical
        super().__init__(imputer, global_transforms)

    def __call__(self, image_chunks):
        transformed_chunks = copy.copy(image_chunks)
        # apply the per-image transforms
        for i, lbl in enumerate(image_chunks):
            for t in self.image_transforms:
                transformed_chunks[lbl] = t[i](transformed_chunks[lbl])

        # concatenate and floating point
        x = build_feature_vector(transformed_chunks, self.is_categorical)
        x = super().__call__(x)
        return x
