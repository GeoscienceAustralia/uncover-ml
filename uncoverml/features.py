import logging

import numpy as np

from uncoverml import mpiops
from uncoverml.image import Image
from uncoverml import patch

log = logging.getLogger(__name__)


def extract_subchunks(image_source, subchunk_index, n_subchunks, patchsize):
    equiv_chunks = n_subchunks * mpiops.chunks
    equiv_chunk_index = n_subchunks * mpiops.chunk_index + subchunk_index
    image = Image(image_source, equiv_chunk_index,
                  equiv_chunks, patchsize)
    x = patch.all_patches(image, patchsize)
    return x


def _image_has_targets(y_min, y_max, im):
    encompass = im.ymin <= y_min and im.ymax >= y_max
    edge_low = im.ymax >= y_min and im.ymax <= y_max
    edge_high = im.ymin >= y_min and im.ymin <= y_max
    inside = encompass or edge_low or edge_high
    return inside


def _extract_from_chunk(image_source, targets, chunk_index, total_chunks,
                        patchsize):
    image_chunk = Image(image_source, chunk_index, total_chunks, patchsize)
    # figure out which chunks I need to consider
    y_min = targets.positions[0, 1]
    y_max = targets.positions[-1, 1]
    if _image_has_targets(y_min, y_max, image_chunk):
        x = patch.patches_at_target(image_chunk, patchsize, targets)
    else:
        x = None
    return x


def extract_features(image_source, targets, n_subchunks, patchsize):
    """
    each node gets its own share of the targets, so all nodes
    will always have targets
    """
    equiv_chunks = n_subchunks * mpiops.chunks
    x_all = []
    for i in range(equiv_chunks):
        x = _extract_from_chunk(image_source, targets, i, equiv_chunks,
                                patchsize)
        if x is not None:
            x_all.append(x)
    if len(x_all) > 0:
        x_all = np.ma.concatenate(x_all, axis=0)
    else:
        raise ValueError("All targets lie outside image boundaries")
    assert x_all.shape[0] == targets.observations.shape[0]
    return x_all


def transform_features(feature_sets, transform_sets, final_transform, config):
    # apply feature transforms
    transformed_vectors = [t(c) for c, t in zip(feature_sets, transform_sets)]
    # TODO remove this when cubist gets removed
    if config.cubist or config.multicubist:
        log.warning("Cubist: ignoring preprocessing transform")
        names = [k for ec in feature_sets for k in ec]
        # 0 is ordinal 1 is categorical
        flags = [int(k.is_categorical) for k in transform_sets]
        feature = [np.zeros(v.shape[1]) + f
                   for v, f in zip(transformed_vectors, flags)]
        feature_vec = {k: v for k, v in zip(names, np.concatenate(feature))}
        config.algorithm_args['feature_type'] = feature_vec
    else:
        feature_vec = None

    x = np.ma.concatenate(transformed_vectors, axis=1)
    if final_transform and not config.cubist:
        x = final_transform(x)
    return x


def gather_features(x, node=None):
    if node:
        x_all = np.ma.vstack(mpiops.comm.gather(x, root=node))
    else:
        x_all = np.ma.vstack(mpiops.comm.allgather(x))
    return x_all


def remove_missing(x, targets=None):
    log.info("Stripping out missing data")
    classes = targets.observations if targets else None
    if np.ma.count_masked(x) > 0:
        no_missing_x = np.sum(x.mask, axis=1) == 0
        x = x.data[no_missing_x]
        # remove labels that correspond to data missing in x
        if targets is not None:
            no_missing_y = no_missing_x[0:(classes.shape[0])]
            classes = classes[:, np.newaxis][no_missing_y]
            classes = classes.flatten()
    else:
        x = x.data

    return x, classes


