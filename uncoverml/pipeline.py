import logging

import numpy as np
from scipy.stats import norm

from revrand.metrics import mll, msll

import uncoverml.defaults as df
from uncoverml import mpiops
from uncoverml import patch
from uncoverml import stats
from uncoverml import validation
from uncoverml.image import Image
from uncoverml.models import modelmaps, apply_multiple_masked, apply_masked


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

        y = targets.observations
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


def validate(targets, data_vectors, cvindex):
    cvind = targets.folds
    Y = targets.observations

    s_ind = np.where(cvind == cvindex)[0]
    t_ind = np.where(cvind != cvindex)[0]

    Yt = Y[t_ind]
    Ys = Y[s_ind]
    Ns = len(Ys)

    # Remove missing data
    data_vectors = [x for x in data_vectors if x is not None]
    EYs = np.ma.concatenate(data_vectors, axis=0)

    # See if this data is already subset for xval
    if len(EYs) > Ns:
        EYs = EYs[s_ind]

    scores = {}
    for m in validation.metrics:

        if m not in validation.probscores:
            score = apply_multiple_masked(validation.score_first_dim(
                                          validation.metrics[m]),
                                          (Ys, EYs))
        elif EYs.ndim == 2:
            if m == 'mll' and EYs.shape[1] > 1:
                score = apply_multiple_masked(mll, (Ys, EYs[:, 0], EYs[:, 1]))
            elif m == 'msll' and EYs.shape[1] > 1:
                score = apply_multiple_masked(msll, (Ys, EYs[:, 0], EYs[:, 1]),
                                              (Yt,))
            else:
                continue
        else:
            continue

        scores[m] = score
        log.info("{} score = {}".format(m, score))

    return scores, Ys, EYs


def create_image(x, shape, bbox, name, outputdir,
                 rgb=True, separatebands=True, band=None):

    # affine
    A, _, _ = geoio.bbox2affine(bbox[1, 0], bbox[0, 0],
                                bbox[0, 1], bbox[1, 1], *shape)

    x_min = None
    x_max = None
    if rgb is True:
        x_min_local = np.ma.min(x, axis=0)
        x_max_local = np.ma.max(x, axis=0)
        x_min = mpiops.comm.allreduce(x_min_local, op=mpiops.min0_op)
        x_max = mpiops.comm.allreduce(x_max_local, op=mpiops.max0_op)

    f = partial(image.to_image_transform, rows=shape[0], x_min=x_min,
                x_max=x_max, band=band, separatebands=separatebands)

    images = f(x)

    # Couple of pieces of information we need here
    if mpiops.chunk_index != 0:
        reqs = []
        for img_idx in range(len(images)):
            reqs.append(mpiops.comm.isend(
                images[img_idx], dest=0, tag=img_idx))
        for r in reqs:
            r.wait()
    else:
        n_images = len(images)
        dtype = images[0].dtype
        n_bands = images[0].shape[2]

        for img_idx in range(n_images):
            band_num = img_idx if band is None else band
            output_filename = os.path.join(outputdir, name +
                                           "_band{}.tif".format(band_num))

            with rasterio.open(output_filename, 'w', driver='GTiff',
                               width=shape[0], height=shape[1],
                               dtype=dtype, count=n_bands, transform=A) as f:
                ystart = 0
                for node in range(mpiops.chunks):
                    data = mpiops.comm.recv(source=node, tag=img_idx) \
                        if node != 0 else images[img_idx]
                    data = np.ma.transpose(data, [2, 1, 0])  # untranspose
                    yend = ystart + data.shape[1]  # this is Y
                    window = ((ystart, yend), (0, shape[0]))
                    index_list = list(range(1, n_bands + 1))
                    f.write(data, window=window, indexes=index_list)
                    ystart = yend


