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
from functools import partial

import uncoverml.defaults as df
from uncoverml import geoio, parallel, feature
from uncoverml.models import apply_masked

log = logging.getLogger(__name__)


# Apply the prediction to the data
def predict(data, model):

    # Ask for predictive outputs if predictive model
    if 'predict_proba' in dir(model):
        predproba = lambda x: np.vstack(model.predict_proba(x)).T
        if 'entropy_reduction' in dir(model):
            pred = lambda x: \
                np.hstack((predproba(x),
                           model.entropy_reduction(x)[:, np.newaxis]))
        else:
            pred = predproba
    else:
        pred = lambda x: model.predict(x)[:, np.newaxis]

    return apply_masked(pred, data)


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--predictname', type=str, default="predicted",
           help="The name to give the predicted target variable.")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use",
           default=None)
@cl.argument('model', type=cl.Path(exists=True))
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(model, files, outputdir, ipyprofile, predictname, quiet):
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
    eff_shape, eff_bbox = feature.load_attributes(filename_dict)

    # Define the transform function to build the features
    cluster = parallel.direct_view(ipyprofile, nchunks)

    # Load the data into a dict on each client
    # Note chunk_indices is a global with different value on each node
    cluster.push({"filename_dict": filename_dict})
    cluster.execute("data_dict = feature.load_data(filename_dict, "
                    "chunk_indices)")

    # Prediction
    f = partial(predict, model=model)

    cluster.push({"f": f, "featurename": predictname, "outputdir": outputdir,
                  "shape": eff_shape, "bbox": eff_bbox})
    cluster.execute("parallel.write_data(data_dict, f, featurename,"
                    "outputdir, shape, bbox)")
