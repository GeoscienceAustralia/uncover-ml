"""
Run a cross-validation metric on a model prediction

.. program-output:: validatemodel --help
"""
import logging
import sys
import os.path
import click as cl

from sklearn.metrics import r2_score
from revrand.validation import smse  # , mll, msll

import uncoverml.defaults as df
from uncoverml import geoio, feature
from uncoverml.validation import input_cvindex, input_targets

log = logging.getLogger(__name__)

metrics = {'r2_score': r2_score,
           'smse': smse,
           }


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--metric', type=cl.Choice(list(metrics.keys())),
           default='r2_score', help="Metrics for scoring prediction quality.")
@cl.argument('cvindex', type=(cl.Path(exists=True), int))
@cl.argument('targets', type=cl.Path(exists=True))
@cl.argument('prediction_files', type=cl.Path(exists=True), nargs=-1)
def main(cvindex, targets, prediction_files, metric, quiet):
    """ Run a cross-validation metric on a model prediction. """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read cv index and targets
    cv_mask = input_cvindex(cvindex[0]) == cvindex[1]
    Ys = input_targets(targets)[cv_mask]

    # build full filenames
    full_filenames = [os.path.abspath(f) for f in prediction_files]
    log.debug("Input files: {}".format(full_filenames))

    # verify the files are all present
    files_ok = geoio.file_indices_okay(full_filenames)
    if not files_ok:
        log.fatal("Input file indices invalid!")
        sys.exit(-1)

    # Load all prediction files
    filename_dict = geoio.files_by_chunk(full_filenames)
    pred_dict = feature.load_data(filename_dict, range(len(filename_dict)))
    EY = feature.data_vector(pred_dict).data  # TODO: deal with missing values
    EYs = EY[cv_mask] if len(EY) > len(Ys) else EY

    log.info("{} score = {}".format(metric, metrics[metric](Ys.flatten(),
                                                            EYs.flatten())))
