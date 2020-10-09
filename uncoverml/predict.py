import logging
from itertools import compress, chain
import numpy as np
import csv
from sklearn.ensemble import BaseEnsemble
from scipy.stats import norm
import pickle

from uncoverml import features
from uncoverml import mpiops
from uncoverml import geoio
from uncoverml.models import apply_masked, modelmaps
from uncoverml.optimise.models import transformed_modelmaps
from uncoverml.krige import krig_dict
from uncoverml import transforms

_logger = logging.getLogger(__name__)
float32finfo = np.finfo(dtype=np.float32)
modelmaps.update(transformed_modelmaps)
modelmaps.update(krig_dict)


def predict(data, model, interval=0.95, **kwargs):

    # Classification
    if hasattr(model, 'predict_proba'):
        def pred(X):
            y, p = model.predict_proba(X)
            predres = np.hstack((y[:, np.newaxis], p))
            return predres

    # Regression
    else:
        def pred(X):
            if hasattr(model, 'predict_dist'):
                Ey, Vy, ql, qu = model.predict_dist(X, interval, **kwargs)
                predres = np.column_stack((Ey, Vy, ql, qu))
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


def _load_mask(subchunk, config):
    image_source = geoio.RasterioImageSource(config.mask)
    return features.extract_subchunks(image_source, subchunk,
                                      config.n_subchunks, config.patchsize)


def _fix_for_corrupt_data(x, feature_names):
    """
    Address this error during prediction:
    "Input contains NaN, infinity or a value too large for dtype('float32')."

    Also address float64 to float32 conversion issue here.
    Several sklearn algorithms performs a data validation and wants float32
    data type.

    From the scikit learn docs:
    "All decision trees use np.float32 arrays internally.
    If training data is not in this format, a copy of the dataset will be made."

    The problem is if our input data is contains values greater than the abs(
    max/min(float32)), then the return value in the float64 to float32
    typecasting is infinity. In other words, for numbers outside float32 max/min
    limits, operations like (np.float64).astype(np.float32) results in
    infinite values.

    See the following comment in stackoverflow.
    https://stackoverflow.com/a/11019850/3321542.

    1. The fix is applied as additional masked areas where nan values are
    found that are not already masked.
    2. Where float32 conversion problem occurs, we limit the data to max/min
    float32, and additionally mask these areas. Just masking was not enough.

    Warns about covariates that cause issues.
    """
    x_isnan = np.isnan(x)

    if x_isnan.any():
        _logger.warning(":mpi:Nans were found in unmasked areas."
                        "Check your covariates for possible nodatavalue, datatype,"
                        "min/max values.")
        problem_feature_names = \
            list(compress(feature_names, x_isnan.any(axis=0)))
        _logger.warning(":mpi: These features were found to be have problems with "
                        "nans present:\n{}".format(problem_feature_names))

    x.mask += x_isnan

    isfinite = np.isfinite(x.astype(np.float32))

    if isfinite.all():
        return x
    else:
        min_mask = x.data < float32finfo.min
        max_mask = x.data > float32finfo.max
        x.data[min_mask] = float32finfo.min
        x.data[max_mask] = float32finfo.max
        x.mask += min_mask + max_mask
        problem_feature_names = \
            list(compress(feature_names, ~isfinite.all(axis=0)))

        _logger.warning(":mpi:Float64 data was truncated to float32 max/min values."
                        "Check your covariates for possible nodatavalue, datatype, "
                        "min/max values.")
        _logger.warning(":mpi:These features were found to be have problems with "
                        "float32 conversion:\n{}".format(problem_feature_names))
    return x


def _get_data(subchunk, config):
    transform_sets = [k.transform_set for k in config.feature_sets]
    extracted_chunk_sets = geoio.image_subchunks(subchunk, config)
    _logger.info("Applying feature transforms")
    x = features.transform_features(extracted_chunk_sets, transform_sets,
                                    config.final_transform, config)[0]
    features_names = geoio.feature_names(config)
    # only check/correct float32 conversion for Ensemble models
    if not config.clustering:
        if (isinstance(modelmaps[config.algorithm](), BaseEnsemble) or
                config.multirandomforest):
            x = _fix_for_corrupt_data(x, features_names)

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
        lat_data = _impute_lat_lon(config.lon_lat['lat'], subchunk, config)
        lon_data = _impute_lat_lon(config.lon_lat['lon'], subchunk, config)
        lon_lat = np.ma.hstack((lon_data, lat_data))
        return _mask_rows(lon_lat, subchunk, config)


def _mask_rows(x, subchunk, config):
    if config.mask:
        mask = _load_mask(subchunk, config)
        mask = mask.reshape(mask.shape[0], 1)
        mask.mask = mask.data != config.retain
        all_mask = np.ma.vstack(mpiops.comm.allgather(mask))
        if all(all_mask.mask):
            x = np.ma.zeros((x.shape[0], len(geoio.feature_names(config))),
                            dtype=np.bool)
            x.mask = True
            _logger.info('Partition {} covariates are not loaded as '
                     'the partition is entirely masked.'.format(subchunk + 1))
            return x
        else:
            _logger.info('Areas with mask={} will be predicted'.format(config.retain))
            mask_x = mask.data[:, 0] != config.retain
            if x.shape[0] != mask.shape[0]:
                raise ValueError(
                    f"Covariate images and provided mask are different sizes (mask shape axis 0: "
                    f"{mask.shape[0]}, image shape axis 0: {x.shape[0]}). Solutions: disable the "
                    f"mask by removing it from the config, or provide a mask that fits the images, "
                    f"or use the cropping 'extents' parameter to crop all data to the smaller of "
                    f"the mask or image size."
                )
            x.mask += np.tile(mask_x, (x.shape[1], 1)).T
    return x

def shapefile_prediction(config, model):
    feature_chunks, positions = features.features_from_shapefile(config.feature_sets)
    transforms = [fs.transform_set for fs in config.feature_sets]
    # Don't need second return val 'keep' mask
    x, _ = features.transform_features(feature_chunks, transforms, 
                                       config.final_transform, config)
    _logger.info(f":mpi:Predicting on {x.shape[0]} samples across {x.shape[1]} features...")
    y_star = predict(x, model, interval=config.quantiles,
                     lon_lat=positions,
                     bootstrap_predictions=config.bootstrap_predictions)

    geoio.write_shapefile_prediction(y_star, model.get_predict_tags(), positions, config)
    
def render_partition(model, subchunk, image_out, config):
    x, feature_names = _get_data(subchunk, config)
    total_gb = mpiops.comm_world.allreduce(x.nbytes / 1e9)
    _logger.info("Loaded {:2.4f}GB of image data".format(total_gb))
    alg = config.algorithm
    _logger.info("Predicting targets for {}.".format(alg))
    
    y_star = predict(x, model, interval=config.quantiles,
                     lon_lat=_get_lon_lat(subchunk, config),
                     bootstrap_predictions=config.bootstrap_predictions)

    if config.clustering and config.cluster_analysis:
        cluster_analysis(x, y_star, subchunk, config, feature_names)
    # cluster_analysis(x, y_star, subchunk, config, feature_names)
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
    _logger.info('Writing cluster analysis results for '
             'partition {}'.format(partition_no))
    mode = 'w' if partition_no == 0 else 'a'

    if mpiops.leader_world:
        with open('cluster_contributions.csv', mode) as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            if partition_no == 0:
                writer.writerow(['feature_names'] + feature_names)
                means = []
                sds = []
                # add feature transform contributions
                for f in config.feature_sets:
                    for t in f.transform_set.global_transforms:
                        means += list(t.mean)
                        sds += list(t.sd)
                # check final transform <- oh god what???
                if config.final_transform is not None:
                    pass
                writer.writerow(['transform mean'] + [str(m) for m in means])
                writer.writerow(['transform sd'] + [str(s) for s in sds])
            writer.writerow(['partition {}'.format(partition_no + 1)])
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

        if mpiops.leader_world:
            writer.writerow(['count-{}'.format(c+1)] + list(class_count))
            writer.writerow(['mean-{}'.format(c+1)] + list(class_mean))
            writer.writerow(['sd-{}'.format(c+1)] + list(sd))


def _flotify(arr):
    return np.array([float(i) for i in arr])


def final_cluster_analysis(n_classes, n_paritions):

    _logger.info('Performing final cluster analysis')

    with open('cluster_contributions.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        names = next(reader)
        means = _flotify(next(reader)[1:])
        stds = _flotify(next(reader)[1:])
        n_covariates = len(names[1:])
        class_pixels = np.zeros(shape=n_classes, dtype=np.int32)
        class_means = np.zeros(shape=(n_classes, n_covariates))
        class_stds = np.zeros(shape=(n_classes, n_covariates))

        for p in range(n_paritions):
            next(reader)  # skip partition string
            for c in range(n_classes):
                count = int(next(reader)[1])  # class count
                class_pixels[c] += count
                if count:  # when there is contribution from this class
                    t_means = _flotify(next(reader)[1:])
                    t_stds = _flotify(next(reader)[1:])
                else:
                    next(reader)  # skip means
                    next(reader)  # skip stds
                    t_means = np.zeros(n_covariates)
                    t_stds = np.zeros(n_covariates)
                class_means[c, :] += count * t_means  # class mean sum
                class_stds[c, :] += count * t_stds * t_stds  # class std sum

    for c in range(n_classes):
        class_means[c] = stds*class_means[c, :]/class_pixels[c] + means
        class_stds[c] = stds*np.sqrt((class_stds[c, :]/class_pixels[c]))

    with open('cluster_contributions_final.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(names)
        for c in range(n_classes):
            writer.writerow(['count-{}'.format(c+1)] + [str(class_pixels[c])])
            writer.writerow(['mean-{}'.format(c+1)] + list(class_means[c]))
            writer.writerow(['sd-{}'.format(c+1)] + list(class_stds[c]))
