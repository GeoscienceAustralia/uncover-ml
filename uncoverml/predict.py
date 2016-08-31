import logging

import numpy as np

from uncoverml import features
from uncoverml.models import apply_masked

log = logging.getLogger(__name__)


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


def render_partition(model, subchunk, image_out, config):

        extracted_chunk_sets = features.image_subchunks(subchunk, config)
        transform_sets = [k.transform_set for k in config.feature_sets]
        x = features.transform_features(extracted_chunk_sets, transform_sets,
                                        config.final_transform)
        alg = config.algorithm
        log.info("Predicting targets for {}.".format(alg))

        y_star = predict(x, model, interval=config.quantiles)
        image_out.write(y_star, subchunk)
