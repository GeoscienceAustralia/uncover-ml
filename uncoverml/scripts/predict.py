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
from uncoverml.models import probmodels, apply_masked

log = logging.getLogger(__name__)


# Apply the prediction to the data
def predict(data, model):

    # Ask for predictive outputs if predictive model
    if isinstance(model, probmodels):
        predwrap = lambda x: np.vstack(model.predict(x, uncertainty=True)).T
    else:
        predwrap = lambda x: model.predict(x)

    return apply_masked(predwrap, data)


# Apply expeded reduction in entropy to data
def entropy_reduct(data, model):

    return apply_masked(model.entropy_reduction, data)


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--predictname', type=str, default="predicted",
           help="The name to give the predicted target variable.")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--entropred', is_flag=True, help="Calculate expected reduction in "
           "entropy, for probabilistic regressors only. The generates another "
           "set of files with 'entropred_' prepended to the output files")
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use",
           default=None)
@cl.argument('model', type=cl.Path(exists=True))
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(model, files, outputdir, ipyprofile, predictname, entropred, quiet):
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
    f = partial(predict, model=model)
    parallel.apply_and_write(cluster, f, "x", predictname, outputdir,
                             eff_shape, eff_bbox)

    # Expected entropy reduction
    if entropred:
        if not isinstance(model, probmodels):
            log.fatal("Cannot calculate expected entropy reduction for"
                      " non-probabilistic models!")
            sys.exit(-1)

        f = partial(entropy_reduct, model=model)
        parallel.apply_and_write(cluster, f, "x", predictname, outputdir,
                                 eff_shape, eff_bbox)
