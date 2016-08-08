import logging

import numpy as np
from scipy.stats import norm

from uncoverml import mpiops, patch, stats
import uncoverml.defaults as df
from uncoverml.image import Image
from uncoverml.models import apply_masked, apply_multiple_masked, modelmaps
from uncoverml.validation import calculate_validation_scores


log = logging.getLogger(__name__)


def extract_transform(x, x_sets):
    x = x.reshape(x.shape[0], -1)
    if x_sets:
        x = stats.one_hot(x, x_sets)
    x = x.astype(float)
    return x


def extract_subchunks(image_source, subchunk_index, n_subchunks, settings):
    equiv_chunks = n_subchunks * mpiops.chunks
    equiv_chunk_index = n_subchunks * mpiops.chunk_index + subchunk_index
    image = Image(image_source, equiv_chunk_index,
                  equiv_chunks, settings.patchsize)
    x = patch.load(image, settings.patchsize, targets=None)
    x = extract_transform(x, settings.x_sets)
    return x


def extract_features(image_source, targets, settings):

    image = Image(image_source, mpiops.chunk_index,
                  mpiops.chunks, settings.patchsize)

    x = patch.load(image, settings.patchsize, targets)

    if settings.onehot and not settings.x_sets:
        settings.x_sets = mpiops.compute_unique_values(x, df.max_onehot_dims)

    if x is not None:
        x = extract_transform(x, settings.x_sets)

    return x, settings


def compose_features(x, settings):
    # verify the files are all present
    x, settings = mpiops.compose_transform(x, settings)
    return x, settings


class CrossvalInfo:
    def __init__(self, scores, y_true, y_pred):
        self.scores = scores
        self.y_true = y_true
        self.y_pred = y_pred


def learn_model(X, targets, algorithm, algorithm_params=None):
    model = modelmaps[algorithm](**algorithm_params)
    y = targets.observations

    if mpiops.chunk_index == 0:
        apply_multiple_masked(model.fit, (X, y))
    model = mpiops.comm.bcast(model, root=0)
    return model


def cross_validate(X, targets, algorithm, algorithm_params=None):
    model = modelmaps[algorithm](**algorithm_params)
    y = targets.observations
    cv_indices = targets.folds

    # Split folds over workers
    fold_list = np.arange(targets.nfolds)
    fold_node = np.array_split(fold_list, mpiops.chunks)[mpiops.chunk_index]

    y_pred = {}
    y_true = {}
    fold_scores = {}

    # Train and score on each fold
    for fold in fold_node:

        train_mask = cv_indices != fold
        test_mask = ~ train_mask

        y_k_train = y[train_mask]
        apply_multiple_masked(model.fit, (X[train_mask], y_k_train))
        y_k_pred = predict(X[test_mask], model)
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
        y_true = np.concatenate([y_true[i] for i in range(targets.nfolds)])
        y_pred = np.concatenate([y_pred[i] for i in range(targets.nfolds)])
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


def predict_channels(model, interval):
    nchannels = 1

    if hasattr(model, 'predict_proba'):
        nchannels += 3 if interval is not None else 1

    if hasattr(model, 'entropy_reduction'):
        nchannels += 1

    return nchannels


def predict(data, model, interval=None):

    def pred(X):

        if hasattr(model, 'predict_proba'):
            Ey, Vy = model.predict_proba(X)
            predres = np.hstack((Ey[:, np.newaxis], Vy[:, np.newaxis]))

            if interval is not None:
                ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))
                predres = np.hstack((predres, ql[:, np.newaxis],
                                     qu[:, np.newaxis]))

            if hasattr(model, 'entropy_reduction'):
                H = model.entropy_reduction(X)
                predres = np.hstack((predres, H[:, np.newaxis]))

        else:
            predres = model.predict(X).flatten()[:, np.newaxis]

        return predres

    return apply_masked(pred, data)
