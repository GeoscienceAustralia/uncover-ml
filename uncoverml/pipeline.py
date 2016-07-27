
import json

import logging

import numpy as np

from scipy.stats import norm

from uncoverml import mpiops, patch, stats

import uncoverml.defaults as df

from uncoverml.image import Image

from uncoverml.models import apply_masked, apply_multiple_masked, modelmaps

from uncoverml.validation import calculate_validation_scores, metrics


log = logging.getLogger(__name__)


def extract_transform(x, x_sets):
    x = x.reshape(x.shape[0], -1)
    if x_sets:
        x = stats.one_hot(x, x_sets)
    x = x.astype(float)
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


def learn_model(X, y, algorithm, cv_indices=None, algorithm_params=None):

    def train(model, X, y, index_mask=None):

        # Remove the rows in this fold
        X_fold = X[index_mask] if index_mask else X
        y_fold = y[index_mask] if index_mask else y

        # Train the model for this fold
        apply_multiple_masked(model.fit, (X_fold, y_fold))

    # Determine the required algorithm and parse it's options
    if algorithm not in modelmaps:
        exit("Invalid algorthm specified")
    args = json.loads(algorithm_params) if algorithm_params is not None else {}

    # Train the master model and store it
    model = modelmaps[algorithm](**args)
    train(model, X, y)
    models = dict()
    models['master'] = model

    # Train the cross validation models if necessary
    if cv_indices is not None:

        # Populate the validation indices
        models['cross_validation'] = []
        models['cv_indices'] = cv_indices

        # Train each model and store it
        for fold in range(max(cv_indices) + 1):

            # Train a model for each row
            remaining_rows = [cv_indices != fold]
            model = modelmaps[algorithm](**args)
            train(model, X, y, remaining_rows)

            # Store the model parameters
            models['cross_validation'].append(model)

    return models


def predict(data, model, interval):

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


def validate(X, y, models, cv_indices):

    # Get all of the cross validation models
    cv_models = models['cross_validation']
    cv_indices = models['cv_indices']

    # Use the models to determine the predicted y's
    y_true = None
    y_pred = None
    score_sum = {m: 0 for m in metrics.keys()}
    for k, model in enumerate(cv_models):

        # Perform the prediction for the Kth index
        y_k_test = y[cv_indices == k]
        y_k_train = y[cv_indices != k]
        y_k_pred = predict(X, model)[cv_indices == k, :]

        # Store the reordered versions for the y-y plot
        y_true = y_k_test if k == 0 else np.append(y_true, y_k_test)
        y_pred = y_k_pred if k == 0 else np.append(y_pred, y_k_pred, axis=0)

        # Use the expected y's to display the validation scores
        scores = calculate_validation_scores(y_k_test,
                                             y_k_train,
                                             y_k_pred)
        score_sum = {m: score_sum[m] + score for (m, score) in scores.items()}

    # Average the scores from each test and store them
    folds = len(cv_models)
    scores = {key: score / folds for key, score in score_sum.items()}

    return scores, y_true, y_pred
