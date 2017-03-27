import logging

import numpy as np

from uncoverml import features
from uncoverml import mpiops
from uncoverml import geoio
from uncoverml.models import apply_masked
from uncoverml import transforms

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
            kr = model.krige_residual(lon_lat=kwargs['lon_lat'])
            predres = np.hstack((predres, kr[:, np.newaxis]))

        if hasattr(model, 'ml_prediction'):
            ml_pred = model.ml_prediction(X)
            predres = np.hstack((predres, ml_pred[:, np.newaxis]))

        return predres
    result = apply_masked(pred, data)
    return result


def _get_data(subchunk, config):
    features_names = geoio.feature_names(config)
    extracted_chunk_sets = geoio.image_subchunks(subchunk, config)
    transform_sets = [k.transform_set for k in config.feature_sets]
    log.info("Applying feature transforms")
    x = features.transform_features(extracted_chunk_sets, transform_sets,
                                    config.final_transform, config)[0]
    return _mask_rows(x, subchunk, config), features_names


def _get_lon_lat(subchunk, config):
    def _impute_lat_lon(cov_file, subchunk, config):
        cov = geoio.RasterioImageSource(cov_file)
        cov_data = features.extract_subchunks(cov, subchunk,
                                              config.n_subchunks,
                                              config.patchsize)
        nn_imputer = transforms.NearestNeighboursImputer()
        cov_data = nn_imputer(cov_data.reshape(cov_data.shape[0], 1))
        return cov_data
    if config.lon_lat:
        lat_data = _impute_lat_lon(config.lat, subchunk, config)
        lon_data = _impute_lat_lon(config.lon, subchunk, config)
        lon_lat = np.ma.hstack((lon_data, lat_data))
        return _mask_rows(lon_lat, subchunk, config)


def _mask_rows(x, subchunk, config):
    mask = config.mask
    if mask:
        mask_source = geoio.RasterioImageSource(mask)
        mask_data = features.extract_subchunks(mask_source, subchunk,
                                               config.n_subchunks,
                                               config.patchsize)
        mask_data = mask_data.reshape(mask_data.shape[0], 1)
        mask_x = mask_data.data[:, 0] != config.retain
        log.info('Areas with mask={} will be predicted'.format(config.retain))

        assert x.shape[0] == mask_x.shape[0], 'shape mismatch of ' \
                                              'mask and inputs'
        x.mask = np.tile(mask_x, (x.shape[1], 1)).T
    return x


def render_partition(model, subchunk, image_out, config):

    x, feature_names = _get_data(subchunk, config)
    total_gb = mpiops.comm.allreduce(x.nbytes / 1e9)
    log.info("Loaded {:2.4f}GB of image data".format(total_gb))
    alg = config.algorithm
    log.info("Predicting targets for {}.".format(alg))
    y_star = predict(x, model, interval=config.quantiles,
                     lon_lat=_get_lon_lat(subchunk, config))
    if config.cluster_analysis:
        cluster_analysis(x, y_star, subchunk, config, feature_names)
    image_out.write(y_star, subchunk)


def cluster_analysis(x, y, partition_no, config, feature_names):
    """
    Parameters
    ----------
    x: ndarray
        array of dim (Ns, d)
    y: ndarry
        array of predictions of dimension (Ns, 1)
    partition_no: int
        partition number of the image
    config: config object
    feature_names: list
        list of strings corresponding to ordered feature names

    """
    import csv
    mode = 'w' if partition_no == 0 else 'a'
    with open('cluster_contributions.csv', mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if mpiops.chunk_index == 0:
            if partition_no == 0:
                writer.writerow(feature_names)
            writer.writerow(['partition {}'.format(partition_no)])
        write_mean_and_sd(x, y, writer, config)


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def write_mean_and_sd(x, y, writer, config):
    for c in range(config.n_classes):
        c_index = (y == c)[:, 0]
        c_count = np.ma.sum(c_index)
        if c_count:
            x_class = x[c_index, :]
            x_sum = np.ma.sum(x_class, axis=0)
            x_count = np.ma.count(x_class, axis=0).ravel()
        else:
            x_sum = np.zeros((2, x.shape[1]))  # just a hack
            x_count = np.zeros((2, x.shape[1]), dtype=np.int32)
        class_sum = mpiops.comm.allreduce(x_sum, op=mpiops.sum0_op)
        class_count = mpiops.comm.allreduce(x_count, op=mpiops.sum0_op)
        class_mean = div0(class_sum, class_count)

        if not c_count:
            x_class = class_mean
        delta_c = ((x_class - class_mean) ** 2)
        delta_c_sum = mpiops.comm.allreduce(delta_c, op=mpiops.sum0_op)
        sd = np.sqrt(delta_c_sum/class_count)

        if mpiops.chunk_index == 0:
            writer.writerow(['mean'] + list(class_mean))
            writer.writerow(['sd'] + list(sd))
