import logging
from collections import OrderedDict
import numpy as np
import pickle
import os
from os.path import basename
import copy

from uncoverml import mpiops, patch, transforms, diagnostics
from uncoverml.image import Image
from uncoverml import patch

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
    y_min = np.min(targets.positions[:,1])
    y_max = np.max(targets.positions[:,1])
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
        raise ValueError(f"Attempting to extract features form {image_source._filename} "
                          "but all targets lie outside image boundaries")
    if x_all.shape[0] != targets.observations.shape[0]:
        raise ValueError(f"Number of covariate data points ({x_all.shape[0]}) not equal to target data points ({targets.observations.shape[0]})")
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
        log.warning("{}: Ignoring preprocessing final transform".format(config.algorithm))
    else:
        if final_transform:
            x = final_transform(x)
    return x, cull_all_null_rows(feature_sets)


def save_intersected_features_and_targets(feature_sets, transform_sets, targets, config, 
                                          impute=True):
    """
    This function saves a table of covariate values and the target 
    value intersected at each point. It also contains columns for 
    UID 'index' and a predicted value. 

    If the target shapefile contains an 'index' field, this will be
    used to populate the 'index' column. This is intended to be used
    as a unique ID for each point in post-processing. If no 'index'
    field exists this column will be zero filled.

    The 'prediction' column is for predicted values created during 
    cross-validation. Again, this is for post-processing. It will only
    be populated if cross-validation is run later on. If not, it will
    be zero filled.

    Two files will be output:
        .../output_dir/{name_of_config}_rawcovariates.csv
        .../output_dir/{name_of_config}_rawcovariates_mask.csv

    This function will also optionally output intersected covariates scatter
    plot and covariate correlation matrix plot.
    """
    if config.fields_to_write_to_csv:
        for f in config.fields_to_write_to_csv:
            if f not in targets.fields:
                raise ValueError(f"write_to_csv field '{f}' does not exist in shapefile records")

    transform_sets_mod = []
    cov_names = ['{}_{}'.format(b, basename(k))
                 for ec in feature_sets
                 for k in ec
                 for b in range(ec[k].shape[3])]

    other_names = ['X', 'Y', 'target', 'prediction']

    if config.fields_to_write_to_csv:
        other_names = config.fields_to_write_to_csv + other_names

    header = ','.join(cov_names + other_names)
    mask_header =','.join(cov_names)

    for t in transform_sets:
        imputer = copy.deepcopy(t.imputer) if impute else None
        dummy_transform = transforms.ImageTransformSet(
            image_transforms=None, imputer=imputer,
            global_transforms=None, is_categorical=t.is_categorical)
        transform_sets_mod.append(dummy_transform)

    transformed_vectors = [t(c) for c, t in zip(feature_sets,
                                                transform_sets_mod)]

    x = np.ma.concatenate(transformed_vectors, axis=1)
    x_all = gather_features(x, node=0)
    all_xy = mpiops.comm.gather(targets.positions, root=0)
    all_targets = mpiops.comm.gather(targets.observations, root=0)

    if config.fields_to_write_to_csv:
        if config.target_search:
            raise NotImplementedError(
                "Can't write 'write_to_csv' columns with target search feature at this time.")
        field_values = []
        for f in config.fields_to_write_to_csv:
            field_values.append(mpiops.comm.gather(targets.fields[f]))

    if mpiops.chunk_index == 0:
        data = [x_all.data]
        if config.fields_to_write_to_csv:
            for f, v  in zip(config.fields_to_write_to_csv, field_values):
                data.append(np.atleast_2d(np.ma.concatenate(v, axis=0)).T)
        all_xy = np.ma.concatenate(all_xy, axis=0)
        all_targets = np.ma.concatenate(all_targets, axis=0)
        xy = np.atleast_2d(all_xy)
        t = np.atleast_2d(all_targets).T
        data += [xy, t]
        # Zeros for prediction values
        data.append(np.zeros(t.shape))
        data = np.hstack(data)

        np.savetxt(config.raw_covariates,
                   X=data, fmt='%s', delimiter=',', header=header, comments='')
        
        np.savetxt(config.raw_covariates_mask,
                   X=~x_all.mask.astype(bool), fmt='%f', delimiter=',', header=mask_header, 
                   comments='')

        if config.plot_intersection:
            diagnostics.plot_covariates_x_targets(
                config.raw_covariates, cols=2).savefig(config.plot_intersection)

        if config.plot_correlation:
            diagnostics.plot_covariate_correlation(
                config.raw_covariates).savefig(config.plot_correlation)

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
    if node is not None:
        x = mpiops.comm.gather(x, root=node)
        if mpiops.chunk_index == node:
            x_all = np.ma.vstack(x)
        else:
            x_all = None
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


