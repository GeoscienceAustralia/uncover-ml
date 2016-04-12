"""
Predict the target values for query data

.. program-output:: predict --help
"""
import logging
import sys
import os.path
import pickle
import click as cl

import uncoverml.defaults as df
from uncoverml import geoio, parallel

log = logging.getLogger(__name__)


# TODO: Get this working with a cvindex file


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
    filename_chunks = geoio.files_by_chunk(full_filenames)
    nchunks = len(filename_chunks)

    # Define the transform function to build the features
    cluster = parallel.direct_view(ipyprofile, nchunks)

    # Load the data into a dict on each client
    # Note chunk_indices is a global with different value on each node
    cluster.push({"chunk_dict": filename_chunks})
    cluster.execute("data = parallel.load_and_cat(chunk_indices, "
                    " chunk_dict)")

    # Apply the prediction to the data
    cluster.push({"model": model, "targetname": predictname,
                  "outputdir": outputdir})
    cluster.execute("parallel.write_predict(data, model, targetname, "
                    "outputdir)")
