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

log = logging.getLogger(__name__)


# Decorator to deal with probabilistic output for non-probabilistic scores
def score_first_dim(func):

    def newscore(y_true, y_pred, *args, **kwargs):

        if y_pred.ndim > 1:
            return func(y_true.flatten(), y_pred[:, 0], *args, **kwargs)
        else:
            return func(y_true.flatten(), y_pred.flatten(), *args, **kwargs)

    return newscore

metrics = {'r2_score': score_first_dim(r2_score),
           'smse': score_first_dim(smse),
           'lins_ccc': score_first_dim(lins_ccc),
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
    pred = feature.data_vector(pred_dict)
    okmask = ~ pred.mask if np.ma.count_masked(pred) != 0 \
        else np.ones(Ns, dtype=bool)

    # See if this data is already subset for xval
    EYs = pred.data[s_ind[okmask]] if len(pred.data) > Ns \
        else pred.data[okmask]

    if metric in nonprob:
        score = metrics[metric](Ys, EYs)
    elif metric == 'mll':
        score = mll(Ys, EYs[:, 0], EYs[:, 1])
    elif metric == 'msll':
        score = msll(Ys, EYs[:, 0], EYs[:, 1], Yt)
    else:
        log.fatal("Invalid metric input")
        sys.exit(-1)

    log.info("{} score = {}".format(metric, score))

    if plotyy:
        maxy = max(Ys.max(), EYs.max())
        pl.plot(Ys, EYs, 'k.')
        pl.plot([0, maxy], [0, maxy], 'r')
        pl.grid(True)
        pl.xlabel('True targets')
        pl.ylabel('Predicted targets')
        pl.title("{} score = {}".format(metric, score))
        pl.show()
