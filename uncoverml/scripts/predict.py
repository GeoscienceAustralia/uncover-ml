"""
Predict the target values for query data

.. program-output:: predict --help
"""
import logging
import sys
import os.path
import pickle
import click as cl
import numpy as np
from scipy.stats import norm
from functools import partial

import uncoverml.defaults as df
from uncoverml import geoio, parallel
from uncoverml.models import apply_masked

log = logging.getLogger(__name__)


# Apply the prediction to the data
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


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--predictname', type=str, default="predicted",
           help="The name to give the predicted target variable.")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use",
           default=None)
@cl.option('--quantiles', type=float, default=None,
           help="Also output quantile intervals for the probabilistic models.")
@cl.argument('model', type=cl.Path(exists=True))
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(model, files, outputdir, ipyprofile, predictname, quantiles, quiet):
    """
    Predict the target values for query data from a machine learning
    algorithm.
    """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # build full filenames
    full_filenames = [os.path.abspath(f) for f in files]
    log.debug("Input files: {}".format(full_filenames))

    # verify the files are all present
    files_ok = geoio.file_indices_okay(full_filenames)
    if not files_ok:
        log.fatal("Input file indices invalid!")
        sys.exit(-1)

    # Load model
    with open(model, 'rb') as f:
        model = pickle.load(f)

    # build the images
    filename_dict = geoio.files_by_chunk(full_filenames)
    nchunks = len(filename_dict)

    # Get the extra hdf5 attributes
    eff_shape, eff_bbox = geoio.load_attributes(filename_dict)

    # Define the transform function to build the features
    cluster = parallel.direct_view(ipyprofile, nchunks)

    # Load the data on each client
    # Note chunk_index is a global with different value on each node
    for i in range(len(cluster)):
        cluster.push({"filenames": filename_dict[i]}, targets=i)
    cluster.execute("x = geoio.load_and_cat(filenames)")

    # Prediction
    f = partial(predict, model=model, interval=quantiles)
    parallel.apply_and_write(cluster, f, "x", predictname, outputdir,
                             eff_shape, eff_bbox)
