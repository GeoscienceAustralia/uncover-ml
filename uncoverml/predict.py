import logging

import numpy as np

from uncoverml import features
from uncoverml import mpiops
from uncoverml import geoio
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


def _get_data(subchunk, config):
    extracted_chunk_sets = geoio.image_subchunks(subchunk, config)
    transform_sets = [k.transform_set for k in config.feature_sets]
    log.info("Applying feature transforms")
    x = features.transform_features(extracted_chunk_sets, transform_sets,
                                    config.final_transform, config)
    return x


def render_partition(model, subchunk, image_out, config):
        x = _get_data(subchunk, config)
        total_gb = mpiops.comm.allreduce(x.nbytes / 1e9)
        log.info("Loaded {:2.4f}GB of image data".format(total_gb))
        alg = config.algorithm
        log.info("Predicting targets for {}.".format(alg))

        y_star = predict(x, model, interval=config.quantiles)
        image_out.write(y_star, subchunk)
