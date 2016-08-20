import logging

import numpy as np

from uncoverml import mpiops, patch, stats
import uncoverml.defaults as df
from uncoverml.image import Image
from uncoverml.models import apply_masked, apply_multiple_masked, modelmaps
from uncoverml.validation import calculate_validation_scores, split_cfold


log = logging.getLogger(__name__)


def extract_subchunks(image_source, subchunk_index, n_subchunks, settings):
    equiv_chunks = n_subchunks * mpiops.chunks
    equiv_chunk_index = n_subchunks * mpiops.chunk_index + subchunk_index
    image = Image(image_source, equiv_chunk_index,
                  equiv_chunks, settings.patchsize)
    x = patch.all_patches(image, settings.patchsize)
    # x = extract_transform(x, settings.x_sets)
    return x


def extract_features(image_source, targets, n_subchunks, patchsize):
    """
    each node gets its own share of the targets, so all nodes
    will always have targets
    """
    equiv_chunks = n_subchunks * mpiops.chunks
    image_chunks = [Image(image_source, k, equiv_chunks, patchsize)
                    for k in range(equiv_chunks)]
    # figure out which chunks I need to consider
    y_min = targets.positions[0, 1]
    y_max = targets.positions[-1, 1]

    def has_targets(im):
        encompass = im.ymin <= y_min and im.ymax >= y_max
        edge_low = im.ymax >= y_min and im.ymax <= y_max
        edge_high = im.ymin >= y_min and im.ymin <= y_max
        inside = encompass or edge_low or edge_high
        return inside

    my_image_chunks = [k for k in image_chunks if has_targets(k)]
    my_x = [patch.patches_at_target(im, patchsize, targets)
            for im in my_image_chunks]
    x = np.ma.concatenate(my_x, axis=0)
    assert x.shape[0] == targets.observations.shape[0]
    # x = extract_transform(x, settings.x_sets)
    return x


def compose_features(x, settings):
    # verify the files are all present
    x, settings = mpiops.compose_transform(x, settings)
    return x, settings


class CrossvalInfo:
    def __init__(self, scores, y_true, y_pred):
        self.scores = scores
        self.y_true = y_true
        self.y_pred = y_pred


def local_learn_model(x_all, targets_all, config):
    model = modelmaps[config.algorithm](**config.algorithm_options)
    y = targets_all.observations

    if mpiops.chunk_index == 0:
        apply_multiple_masked(model.fit, (x_all, y),
                              kwargs={'fields': targets_all.fields})
    model = mpiops.comm.bcast(model, root=0)
    return model


def local_crossval(x_all, targets_all, config):
    model = modelmaps[config.algorithm](**config.algorithm_options)
    y = targets_all.observations
    _, cv_indices = split_cfold(y.shape[0], config.folds, config.crossval_seed)

    # Split folds over workers
    fold_list = np.arange(config.folds)
    fold_node = np.array_split(fold_list, mpiops.chunks)[mpiops.chunk_index]

    y_pred = {}
    y_true = {}
    fold_scores = {}

    # Train and score on each fold
    for fold in fold_node:

        train_mask = cv_indices != fold
        test_mask = ~ train_mask

        y_k_train = y[train_mask]

        # Extra fields
        fields_train = {f: v[train_mask] for f, v in targets_all.fields.items()}
        fields_pred = {f: v[test_mask] for f, v in targets_all.fields.items()}

        # Train on this fold
        apply_multiple_masked(model.fit, data=(x_all[train_mask], y_k_train),
                              kwargs={'fields': fields_train})
        y_k_pred = predict(x_all[test_mask], model, fields=fields_pred)
        y_k_test = y[test_mask]

        y_pred[fold] = y_k_pred
        y_true[fold] = y_k_test

        fold_scores[fold] = calculate_validation_scores(y_k_test,
                                                        y_k_train,
                                                        y_k_pred)
    y_pred = join_dicts(mpiops.comm.gather(y_pred, root=0))
    y_true = join_dicts(mpiops.comm.gather(y_true, root=0))
    scores = join_dicts(mpiops.comm.gather(fold_scores, root=0))
    result = None
    if mpiops.chunk_index == 0:
        y_true = np.concatenate([y_true[i] for i in range(config.folds)])
        y_pred = np.concatenate([y_pred[i] for i in range(config.folds)])
        valid_metrics = scores[0].keys()
        scores = {m: np.mean([d[m] for d in scores.values()])
                  for m in valid_metrics}
        result = CrossvalInfo(scores, y_true, y_pred)
    result = mpiops.comm.bcast(result, root=0)
    return result


def join_dicts(dicts):

    if dicts is None:
        return

    d = {k: v for D in dicts for k, v in D.items()}

    return d


def predict(data, model, interval=0.95, **kwargs):

    def pred(X):

        if hasattr(model, 'predict_proba'):
            Ey, Vy, ql, qu = model.predict_proba(X, interval, **kwargs)
            predres = np.hstack((Ey[:, np.newaxis], Vy[:, np.newaxis],
                                 ql[:, np.newaxis], qu[:, np.newaxis]))

        else:
            predres = np.reshape(model.predict(X, **kwargs),
                                 newshape=(len(X), 1))

        if hasattr(model, 'entropy_reduction'):
            MI = model.entropy_reduction(X)
            predres = np.hstack((predres, MI[:, np.newaxis]))

        return predres

    return apply_masked(pred, data)
