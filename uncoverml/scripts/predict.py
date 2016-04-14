"""
Predict the target values for query data

.. program-output:: predict --help
"""
import logging
import sys
import os.path
import pickle
import click as cl
from functools import partial
# import numpy as np

import uncoverml.defaults as df
from uncoverml import geoio, parallel
# from uncoverml.validation import input_cvindex

log = logging.getLogger(__name__)


# Apply the prediction to the data
def predict(data, model):
    # FIXME deal with missing values (mask)
    return model.predict(data.data)


# TODO: Get this working with a cvindex file


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--cvindex', type=(cl.Path(exists=True), int), default=(None, None),
           help="Optional cross validation index file and index to hold out.")
@cl.option('--predictname', type=str, default="predicted",
           help="The name to give the predicted target variable.")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use",
           default=None)
@cl.argument('model', type=cl.Path(exists=True))
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(model, files, outputdir, ipyprofile, predictname, cvindex, quiet):

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

    # Optionally subset the data for cross validation
    # TODO
    # if cvindex[0] is not None:
    #     cv_chunks = np.array_split(input_cvindex(cvindex[0]), nchunks)

    # Define the transform function to build the features
    cluster = parallel.direct_view(ipyprofile, nchunks)

    # Load the data into a dict on each client
    # Note chunk_indices is a global with different value on each node
    cluster.push({"filename_dict":filename_dict})
    cluster.execute("data_dict = feature.load_data(filename_dict, chunk_indices)")


    f = partial(predict, model=model)

    cluster.push({"f": f, "featurename": predictname, "outputdir": outputdir})
    cluster.execute("parallel.write_data(data_dict, f, featurename, outputdir)")
    sys.exit(0)

    # cluster.push({"model": model, "targetname": predictname,
    #               "outputdir": outputdir})
    # cluster.execute("parallel.write_predict(data, model, targetname, "
    #                 "outputdir)")
