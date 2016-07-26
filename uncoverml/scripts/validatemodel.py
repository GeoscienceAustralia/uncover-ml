
"""
Run cross-validation metrics on a model prediction
.. program-output:: validatemodel --help
"""

import json

import logging

import pickle

import click as cl

import click_log as cl_log

import matplotlib.pyplot as pl

from mpi4py import MPI

from revrand.metrics import lins_ccc, mll, msll, smse

from sklearn.metrics import explained_variance_score, r2_score

from uncoverml.models import apply_multiple_masked

from uncoverml.scripts.learnmodel import load_training_data

from uncoverml.scripts.predict import predict


log = logging.getLogger(__name__)


def get_first_dim(y):
    return y[:, 0] if y.ndim > 1 else y


# Decorator to deal with probabilistic output for non-probabilistic scores
def score_first_dim(func):
    def newscore(y_true, y_pred, *args, **kwargs):
        return func(y_true.flatten(), get_first_dim(y_pred), *args, **kwargs)
    return newscore


metrics = {'r2_score': r2_score,
           'expvar': explained_variance_score,
           'smse': smse,
           'lins_ccc': lins_ccc,
           'mll': mll,
           'msll': msll}


def calculate_validation_scores(ys, yt, eys):

    probscores = ['msll', 'mll']

    scores = {}
    for m in metrics:

        if m not in probscores:
            score = apply_multiple_masked(score_first_dim(metrics[m]),
                                          (ys, eys))
        elif eys.ndim == 2:
            if m == 'mll' and eys.shape[1] > 1:
                score = apply_multiple_masked(mll, (ys, eys[:, 0], eys[:, 1]))
            elif m == 'msll' and eys.shape[1] > 1:
                score = apply_multiple_masked(msll, (ys, eys[:, 0], eys[:, 1]),
                                              (yt,))
            else:
                continue
        else:
            continue

        scores[m] = score
    return scores


def y_y_plot(y1, y2, y_label=None, y_exp_label=None, title=None,
             outfile=None, display=None):
    fig = pl.figure()
    maxy = max(y1.max(), get_first_dim(y2).max())
    miny = min(y1.min(), get_first_dim(y2).min())
    apply_multiple_masked(pl.plot, (y1, get_first_dim(y2)), ('k.',))
    pl.plot([miny, maxy], [miny, maxy], 'r')
    pl.grid(True)
    pl.xlabel(y_label)
    pl.ylabel(y_exp_label)
    pl.title(title)
    if outfile is not None:
        fig.savefig(outfile + ".png")
    if display:
        pl.show()


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
    y_true = []
    y_predicted = []
    score_sum = {m: 0 for m in metrics.keys()}
    for k, model in enumerate(cv_models):

        # Perform the prediction for the Kth index
        y_k_test = y[cv_indices == k]
        y_k_train = y[cv_indices != k]
        y_k_predicted = predict(X, model)[cv_indices == k, :]

        # Store the reordered versions for the y-y plot
        y_true.append(y_k_test)
        y_predicted.append(y_k_predicted)

        # Use the expected y's to display the validation scores
        scores = calculate_validation_scores(y_k_test,
                                             y_k_train,
                                             y_k_predicted)
        score_sum = {m: score_sum[m] + score for (m, score) in scores.items()}

    # Average the scores from each test and store them
    folds = len(cv_models)
    scores = {key: score / folds for key, score in score_sum.items()}

    # Log then output the scores
    for m, score in scores.items():
        log.info("{} score = {}".format(m, score))
    if outfile is not None:
        with open(outfile + ".json", 'w') as f:
            json.dump(scores, f, sort_keys=True, indent=4)

    # Make a figure if necessary
    if plotyy and (outfile is not None):
        y_y_plot(y, y_predicted,
                 title='True vs. predicted target values.',
                 y_label='True targets',
                 y_exp_label='Predicted targets',
                 outfile=outfile,
                 display=True)
