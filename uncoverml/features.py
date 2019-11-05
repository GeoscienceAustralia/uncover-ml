import logging
from collections import OrderedDict
import numpy as np
import pickle
from os.path import basename
import os

from uncoverml import mpiops
from uncoverml.image import Image
from uncoverml import patch
from uncoverml import transforms

log = logging.getLogger(__name__)


def extract_subchunks(image_source, subchunk_index, n_subchunks, patchsize):
    equiv_chunks = n_subchunks * mpiops.chunks
    equiv_chunk_index = mpiops.chunks*subchunk_index + mpiops.chunk_index
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
        x_all_data = np.concatenate([a.data for a in x_all], axis=0)
        x_all_mask = np.concatenate([a.mask for a in x_all], axis=0)
        x_all = np.ma.masked_array(x_all_data, mask=x_all_mask)
    else:
        raise ValueError("All targets lie outside image boundaries")
    assert x_all.shape[0] == targets.observations.shape[0]
    return x_all


def transform_features(feature_sets, transform_sets, final_transform, config):
    # apply feature transforms
    transformed_vectors = [t(c) for c, t in zip(feature_sets, transform_sets)]
    # TODO remove this when cubist gets removed
    if config.cubist or config.multicubist:
        feature_vec = OrderedDict()
        log.warning("Cubist: ignoring preprocessing transform")
        names = ['{}_{}'.format(b, k)
                 for ec in feature_sets
                 for k in ec
                 for b in range(ec[k].shape[3])]
        # 0 is ordinal 1 is categorical
        flags = [int(k.is_categorical) for k in transform_sets]
        feature = [np.zeros(v.shape[1]) + f
                   for v, f in zip(transformed_vectors, flags)]
        for k, v in zip(names, np.concatenate(feature)):
            feature_vec[k] = v
        config.algorithm_args['feature_type'] = feature_vec
        if mpiops.chunk_index == 0 \
                and config.pk_featurevec and not os.path.exists(config.pk_featurevec):
            log.info('Saving featurevec for reuse')
            pickle.dump(feature_vec, open(config.pk_featurevec, 'wb'))

    x = np.ma.concatenate(transformed_vectors, axis=1)
    if config.cubist or config.multicubist or config.krige:
        log.warning("{}: Ignoring preprocessing "
                    "transform".format(config.algorithm))
    else:
        if final_transform:
            x = final_transform(x)
    return x, cull_all_null_rows(feature_sets)


def save_intersected_features_and_targets(feature_sets, transform_sets, targets, config):
    """
    This function saves raw covariates values at the target locations, i.e.,
    after the targets have been intersected.

    This will save the following two files if they are provided in the
    config file:
        a) rawcovariates.csv: the covariate values in csv
        b) rawcovariates_mask.csv: the corresponding mask in csv
    This function will also optionally output intersected covariates scatter
    plot.
    """
    transform_sets_mod = []
    names = ['{}_{}'.format(b, basename(k))
             for ec in feature_sets
             for k in ec
             for b in range(ec[k].shape[3])]

    names += ["X", "Y", config.target_property + "(target)"]
    header = ', '.join(names)

    for t in transform_sets:
        dummy_transform = transforms.ImageTransformSet(
            image_transforms=None, imputer=None,
            global_transforms=None, is_categorical=t.is_categorical)
        transform_sets_mod.append(dummy_transform)

    transformed_vectors = [t(c) for c, t in zip(feature_sets,
                                                transform_sets_mod)]

    x = np.ma.concatenate(transformed_vectors, axis=1)
    x_all = gather_features(x, node=0)

    all_xy = mpiops.comm.gather(targets.positions, root=0)
    all_targets = mpiops.comm.gather(targets.observations, root=0)

    if mpiops.chunk_index == 0:
        all_xy = np.ma.concatenate(all_xy, axis=0)
        all_targets = np.ma.concatenate(all_targets, axis=0)
        xy = np.atleast_2d(all_xy)
        t = np.atleast_2d(all_targets).T
        data = np.hstack((x_all.data, xy, t))
        np.savetxt(config.rawcovariates, X=data, delimiter=',',
                   fmt='%.4e',
                   header=header, comments='')
        mask = np.hstack((x_all.mask.astype(int), np.zeros_like(t)))
        np.savetxt(config.rawcovariates_mask, X=mask,
                   delimiter=',', fmt='%d', header=header, comments='')
        if config.plot_covariates:
            import matplotlib.pyplot as plt
            for i, name in enumerate(names[:-3]):
                log.info('plotting {}'.format(name))
                plt.figure()
                vals = x_all[:, i]
                vals_no_mask = vals[~vals.mask].data
                plt.scatter(x=list(range(vals_no_mask.shape[0])),
                            y=vals_no_mask.data)
                plt.title(name)
                plt.savefig(os.path.join(config.plot_covariates, name.rstrip('.tif') + '.png'))
                plt.close()


def cull_all_null_rows(feature_sets):
    # cull targets with all null values
    dummy_transform = transforms.ImageTransformSet(image_transforms=None,
                                                   imputer=None,
                                                   global_transforms=None,
                                                   is_categorical=True)
    transformed_vectors = [dummy_transform(c) for c in feature_sets]

    bool_transformed_vectors = np.concatenate([t.mask for t in
                                               transformed_vectors], axis=1)
    covaraiates = bool_transformed_vectors.shape[1]
    rows_to_keep = np.sum(bool_transformed_vectors, axis=1) != covaraiates
    return rows_to_keep


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


