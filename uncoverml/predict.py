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

        if hasattr(model, 'krige_residual'):
            MI = model.krige_residual(lon_lat=kwargs['lon_lat'])
            predres = np.hstack((predres, MI[:, np.newaxis]))

        return predres
    result = apply_masked(pred, data)
    return result


def _get_data(subchunk, config):
    extracted_chunk_sets = geoio.image_subchunks(subchunk, config)
    transform_sets = [k.transform_set for k in config.feature_sets]
    log.info("Applying feature transforms")
    x = features.transform_features(extracted_chunk_sets, transform_sets,
                                    config.final_transform, config)[0]
    return _mask_rows(x, subchunk, config)


def _get_lon_lat(subchunk, config):
    if config.lon_lat:
        lat = geoio.RasterioImageSource(config.lat)
        lat_data = features.extract_subchunks(lat, subchunk,
                                              config.n_subchunks, 0)
        lon = geoio.RasterioImageSource(config.lon)
        lon_data = features.extract_subchunks(lon, subchunk,
                                              config.n_subchunks, 0)
        lon_lat = np.hstack((lon_data, lat_data)).reshape(
            (lon_data.shape[0], 2))
        return _mask_rows(lon_lat, subchunk, config)


def _mask_rows(x, subchunk, config):
    mask = config.mask
    if mask:
        mask_source = geoio.RasterioImageSource(mask)
        mask_data = features.extract_subchunks(mask_source, subchunk,
                                               config.n_subchunks,
                                               config.patchsize)
        mask_x = mask_data.data[:, 0, 0, 0] != config.retain
        log.info('Areas with mask={} will be predicted'.format(config.retain))

        assert x.shape[0] == mask_x.shape[0], 'shape mismatch of ' \
                                              'mask and inputs'
        x.mask = np.tile(mask_x, (x.shape[1], 1)).T
    return x


def render_partition(model, subchunk, image_out, config):
    x = _get_data(subchunk, config)
    total_gb = mpiops.comm.allreduce(x.nbytes / 1e9)
    log.info("Loaded {:2.4f}GB of image data".format(total_gb))
    alg = config.algorithm
    log.info("Predicting targets for {}.".format(alg))
    y_star = predict(x, model, interval=config.quantiles,
                     lon_lat=_get_lon_lat(subchunk, config))
    image_out.write(y_star, subchunk)
