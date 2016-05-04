"""
Run a cross-validation metric on a model prediction

.. program-output:: validatemodel --help
"""
import logging
import sys
import os.path
import click as cl
import numpy as np
import matplotlib.pyplot as pl

from sklearn.metrics import r2_score
from revrand.validation import smse, mll, msll

import uncoverml.defaults as df
from uncoverml import geoio, feature
from uncoverml.validation import input_cvindex, input_targets, lins_ccc
from uncoverml.models import apply_multiple_masked

log = logging.getLogger(__name__)


def get_first_dim(Y):

    return Y[:, 0] if Y.ndim > 1 else Y


# Decorator to deal with probabilistic output for non-probabilistic scores
def score_first_dim(func):

    def newscore(y_true, y_pred, *args, **kwargs):

        return func(y_true.flatten(), get_first_dim(y_pred), *args, **kwargs)

    return newscore

metrics = {'r2_score': r2_score,
           'smse': smse,
           'lins_ccc': lins_ccc,
           'mll': mll,
           'msll': msll
           }

nonprob = ['r2_score', 'smse', 'lins_ccc']


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--metric', type=cl.Choice(list(metrics.keys())),
           default='r2_score', help="Metrics for scoring prediction quality")
@cl.option('--plotyy', is_flag=True, help="plot the target vs. prediction")
@cl.argument('cvindex', type=(cl.Path(exists=True), int))
@cl.argument('targets', type=cl.Path(exists=True))
@cl.argument('prediction_files', type=cl.Path(exists=True), nargs=-1)
def main(cvindex, targets, prediction_files, metric, quiet, plotyy):
    """ Run a cross-validation metric on a model prediction. """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read cv index and targets
    cvind = input_cvindex(cvindex[0])
    s_ind = np.where(cvind == cvindex[1])[0]
    t_ind = np.where(cvind != cvindex[1])[0]

    Y = input_targets(targets)
    Yt = Y[t_ind]
    Ys = Y[s_ind]
    Ns = len(Ys)

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

    # Deal with missing data
    EYs = feature.data_vector(pred_dict)

    # See if this data is already subset for xval
    if len(EYs) > Ns:
        EYs = EYs[s_ind]

    if metric in nonprob:
        score = apply_multiple_masked(score_first_dim(metrics[metric]),
                                      (Ys, EYs))
    elif metric == 'mll':
        score = apply_multiple_masked(mll, (Ys, EYs[:, 0], EYs[:, 1]))
    elif metric == 'msll':
        score = apply_multiple_masked(msll, (Ys, EYs[:, 0], EYs[:, 1]), (Yt,))
    else:
        log.fatal("Invalid metric input")
        sys.exit(-1)

    log.info("{} score = {}".format(metric, score))

    if plotyy:
        maxy = max(Ys.max(), get_first_dim(EYs).max())
        miny = min(Ys.min(), get_first_dim(EYs).min())
        apply_multiple_masked(pl.plot, (Ys, get_first_dim(EYs)), ('k.',))
        pl.plot([miny, maxy], [miny, maxy], 'r')
        pl.grid(True)
        pl.xlabel('True targets')
        pl.ylabel('Predicted targets')
        pl.title("{} score = {}".format(metric, score))
        pl.show()
