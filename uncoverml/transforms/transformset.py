import copy
import logging

import numpy as np

from uncoverml import mpiops

log = logging.getLogger(__name__)


# def build_feature_vector(image_chunks):
#     # flatten image patches
#     image_chunks = {k: v.reshape(v.shape[0], -1)
#                     for k, v in image_chunks.items()}

#     # size up x
#     image_0 = next(iter(image_chunks.keys()))
#     n_features = image_chunks[image_0].shape[0]
#     ndims = np.sum([x.shape[1] for x in image_chunks.values()])
#     # create the memory
#     x = np.empty((n_features, ndims), dtype=float)
#     mask = np.empty((n_features, ndims), dtype=bool)
#     # transfer in the data
#     start_idx = 0
#     for k, im in image_chunks.items():
#         ndims_im = im.shape[1]
#         end_idx = start_idx + ndims_im
#         x[:, start_idx:end_idx] = im.data
#         mask[:, start_idx:end_idx] = im.mask
#         start_idx += ndims_im
#     result = np.ma.MaskedArray(data=x, mask=mask)
#     return result


def build_feature_vector(image_chunks):
    for k, im in image_chunks.items():
        image_chunks[k] = im.reshape(im.shape[0], -1)
    x = np.ma.concatenate(image_chunks.values(), axis=1)
    return x


def missing_percentage(x):
    x_n = mpiops.count(x)
    x_full_local = x.shape[0]
    x_full = mpiops.comm.allreduce(x_full_local)
    missing = (1.0 - np.sum(x_n) / (x_full * x_n.shape[0])) * 100.0
    return missing


class TransformSet:
    def __init__(self, imputer=None, transforms=None):
            self.transforms = (transforms if transforms else [])
            self.imputer = imputer

    def __call__(self, x):
        # impute
        if self.imputer:
            missing_percent = missing_percentage(x)
            log.info("Imputing {}% missing data".format(missing_percent))
            x = self.imputer(x)

        # transforms
        for t in self.transforms:
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
