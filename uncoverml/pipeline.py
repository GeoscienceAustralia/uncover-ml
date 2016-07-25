import logging

import numpy as np
from scipy.stats import norm

import uncoverml.defaults as df
from uncoverml import mpiops
from uncoverml import patch
from uncoverml import stats
from uncoverml import geoio
from uncoverml.models import modelmaps, apply_multiple_masked, apply_masked


log = logging.getLogger(__name__)


def extract_transform(x, x_sets):
    x = x.reshape(x.shape[0], -1)
    if x_sets:
        x = stats.one_hot(x, x_sets)
    x = x.astype(float)
    return x


def extract_features(settings, target_infile, geotiff_infile, hdf_outfile):

    # Compute the effective sampled resolution accounting for patchsize
    image_source = geoio.RasterioImageSource(geotiff_infile)
    full_image = geoio.Image(image_source)
    eff_shape = full_image.patched_shape(settings.patchsize)
    eff_bbox = full_image.patched_bbox(settings.patchsize)

    image = geoio.Image(image_source, mpiops.chunk_index,
                        mpiops.chunks, settings.patchsize)

    x = patch.load(image, settings.patchsize, target_infile)

    if settings.onehot and not settings.x_sets:
        settings.x_sets = mpiops.compute_unique_values(x, df.max_onehot_dims)

    if x is not None:
        x = extract_transform(x, settings.x_sets)

    geoio.output_features(x, hdf_outfile, shape=eff_shape, bbox=eff_bbox)

    return settings


def compose_features(settings, hdf_infiles, hdf_outfile):

    # verify the files are all present
    filename_dict = geoio.files_by_chunk(hdf_infiles)

    # Get attribs if they exist
    eff_shape, eff_bbox = geoio.load_attributes(filename_dict)
    chunk_files = filename_dict[mpiops.chunk_index]
    x = geoio.load_and_cat(chunk_files)
    x, settings = mpiops.compose_transform(x, settings)

    geoio.output_features(x, hdf_outfile, shape=eff_shape, bbox=eff_bbox)

    return settings


def learn_model(X_list, targets, algorithm,
                cvindex=None, algorithm_params=None):
    # Remove the missing data
    data_vectors = [x for x in X_list if x is not None]
    X = np.ma.concatenate(data_vectors, axis=0)

    # Optionally subset the data for cross validation
    if cvindex is not None:
        cv_ind = targets.folds

        # TODO: temporary fix!!!! REMOVE THIS
        cv_ind = cv_ind[::-1]

        y = targets.observatinos
        y = y[cv_ind != cvindex]
        X = X[cv_ind != cvindex]

    # Train the model
    mod = modelmaps[algorithm](**algorithm_params)
    apply_multiple_masked(mod.fit, (X, y))
    return mod


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

