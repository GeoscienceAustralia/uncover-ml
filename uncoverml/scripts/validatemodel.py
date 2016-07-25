
import json

import logging

import os.path

import sys

import pickle

import click as cl

import click_log as cl_log

import matplotlib.pyplot as pl

from mpi4py import MPI

import numpy as np

from uncoverml.scripts.predict import predict

from uncoverml.scripts.learnmodel import load_training_data

from revrand.metrics import lins_ccc, mll, msll, smse

from sklearn.metrics import explained_variance_score, r2_score

from uncoverml import geoio

from uncoverml.models import apply_multiple_masked

log = logging.getLogger(__name__)


def get_first_dim(Y):
    return Y[:, 0] if Y.ndim > 1 else Y


# Decorator to deal with probabilistic output for non-probabilistic scores
def score_first_dim(func):
    def newscore(y_true, y_pred, *args, **kwargs):
        return func(y_true.flatten(), get_first_dim(y_pred), *args, **kwargs)
    return newscore


def calculate_validation_scores(Ys, EYs):

    scores = {}
    for m in metrics:

        if m not in probscores:
            score = apply_multiple_masked(score_first_dim(metrics[m]),
                                          (Ys, EYs))
        elif EYs.ndim == 2 and m == 'mll' and EYs.shape[1] > 1:
            score = apply_multiple_masked(mll, (Ys, EYs[:, 0], EYs[:, 1]))
        
        elif EYs.ndim == 2 and m == 'msll' and EYs.shape[1] > 1:
            score = apply_multiple_masked(msll, (Ys, EYs[:, 0], EYs[:, 1]),
                                          (Yt,))
        else:
            continue

        scores[m] = score
        log.info("{} score = {}".format(m, score))
        return scores


def make_y_y_plot(y1, y2, y1_label=None, y2_label=None, title=None, 
    outfile=None):
        fig = pl.figure()
        maxy = max(Ys.max(), get_first_dim(EYs).max())
        miny = min(Ys.min(), get_first_dim(EYs).min())
        apply_multiple_masked(pl.plot, (Ys, get_first_dim(EYs)), ('k.',))
        pl.plot([miny, maxy], [miny, maxy], 'r')
        pl.grid(True)
        pl.xlabel(y1_label)
        pl.ylabel(y2_label)
        pl.title(title)
        if outfile is not None:
            fig.savefig(outfile + ".png")
        if plotyy:
            pl.show()


metrics = {'r2_score': r2_score,
           'expvar': explained_variance_score,
           'smse': smse,
           'lins_ccc': lins_ccc,
           'mll': mll,
           'msll': msll
           }

probscores = ['msll', 'mll']

@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--outfile', type=cl.Path(exists=False), default=None,
           help="File name (minus extension) to save output too")
@cl.option('--plotyy', is_flag=True, help="Show plot of the target vs."
           "prediction, otherwise just save")
@cl.argument('model', type=cl.Path(exists=True))
@cl.argument('targets', type=cl.Path(exists=True))
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(model, targets, files, plotyy, outfile):
    """ Run cross-validation metrics on a model prediction.
    The following metrics are evaluated:
    - R-square
    - Explained variance
    - Standardised Mean Squared Error
    - Lin's concordance correlation coefficient
    - Mean Gaussian negative log likelihood (for probabilistic predictions)
    - Standardised mean Gaussian negative log likelihood (for probabilistic
      predictions)
    """

    # Make sure python only runs on a single machine at a time
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 0:
        return

    # Get all of the input data
    X, y, cv_indices = load_training_data(files, targets)

    # Get all of the cross validation models 
    with open(model, 'rb') as f:
        models = pickle.load(f)
        cv_models = models['cross_validation']
        cv_indices = models['cv_indices']


    # Use the models to determine the predicted y's
    y_pred = np.zeros(len(y))
    for index, model in enumerate(cv_models):
        
        # Perform the prediction
        y_pred = predict(X, model)

        # Only take the predictions that correspond to this model
        y_pred[cv_indices==index] = y[cv_indices==index]

    rank = comm.Get_rank()
    if rank == 0:
        import IPython; IPython.embed()
    comm.barrier()   

    
    # Use the expected y's to display the validation scores
    scores = calculate_validation_scores(Ys, EYs)
    if outfile is not None:
        with open(outfile + ".json", 'w') as f:
            json.dump(scores, f, sort_keys=True, indent=4)

    # Make figure
    if plotyy and (outfile is not None):
        make_y_y_plot( y, y_pred,
              title = 'True vs. predicted target values.',
              y1_label = 'True targets',
              y2_label = 'Predicted targets',
              outfile = outfile)
