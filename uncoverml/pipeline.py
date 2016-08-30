import logging

import numpy as np

from uncoverml import mpiops, patch
from uncoverml.image import Image
from uncoverml.models import apply_masked, apply_multiple_masked, modelmaps
from uncoverml.validation import calculate_validation_scores, split_cfold


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
    x_all = np.ma.concatenate(x_all, axis=0)
    assert x_all.shape[0] == targets.observations.shape[0]
    return x_all


def transform_features(feature_sets, transform_sets, final_transform):
    # apply feature transforms
    transformed_vectors = [t(c) for c, t in zip(feature_sets, transform_sets)]
    x = np.ma.concatenate(transformed_vectors, axis=1)
    x = final_transform(x)
    return x


class CrossvalInfo:
    def __init__(self, scores, y_true, y_pred):
        self.scores = scores
        self.y_true = y_true
        self.y_pred = y_pred


def local_learn_model(x_all, targets_all, config):
    model = modelmaps[config.algorithm](**config.algorithm_args)
    y = targets_all.observations

    if mpiops.chunk_index == 0:
        apply_multiple_masked(model.fit, (x_all, y),
                              kwargs={'fields': targets_all.fields})
    model = mpiops.comm.bcast(model, root=0)
    return model


def local_crossval(x_all, targets_all, config):
    log.info("Validating with {} folds".format(config.folds))
    model = modelmaps[config.algorithm](**config.algorithm_args)
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
        fields_train = {f: v[train_mask]
                        for f, v in targets_all.fields.items()}
        fields_pred = {f: v[test_mask] for f, v in targets_all.fields.items()}

        # Train on this fold
        apply_multiple_masked(model.fit, data=(x_all[train_mask], y_k_train),
                              kwargs={'fields': fields_train})

        # Testing
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
        score_string = "Validation complete:\n"
        for metric, score in scores.items():
            score_string += "{}\t= {}\n".format(metric, score)
        log.info(score_string)

        result_tags = model.get_predict_tags()
        y_pred_dict = dict(zip(result_tags, y_pred.T))
        result = CrossvalInfo(scores, y_true, y_pred_dict)
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
