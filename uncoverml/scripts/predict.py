"""
Predict the target values for query data

.. program-output:: predict --help
"""
import logging
import sys
import os.path
import pickle
import click as cl
import click_log as cl_log
from functools import partial

from uncoverml import mpiops
from uncoverml import pipeline
from uncoverml import geoio

log = logging.getLogger(__name__)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--predictname', type=str, default="predicted",
           help="The name to give the predicted target variable.")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--quantiles', type=float, default=None,
           help="Also output quantile intervals for the probabilistic models.")
@cl.argument('model', type=cl.Path(exists=True))
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(model, files, outputdir, predictname, quantiles):
    """
    Predict the target values for query data from a machine learning algorithm.
    """
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

    x = geoio.load_and_cat(filename_dict[mpiops.chunk_index])

    # Prediction
    f = partial(pipeline.predict, model=model, interval=quantiles)

    outfile = geoio.output_filename(predictname, mpiops.chunk_index,
                                    mpiops.chunks, outputdir)
    if x is not None:
        log.info("Applying final transform and writing output files")
        f_x = f(x)
        geoio.output_features(f_x, outfile)
    else:
        geoio.output_blank(outfile)
