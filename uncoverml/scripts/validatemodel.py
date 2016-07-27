
"""
Run cross-validation metrics on a model prediction
.. program-output:: validatemodel --help
"""

import json

import logging

import pickle

import click as cl

import click_log as cl_log

from uncoverml import mpiops, pipeline

from uncoverml.scripts.learnmodel import load_training_data

from uncoverml.validation import y_y_plot

log = logging.getLogger(__name__)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--outfile', type=cl.Path(exists=False), default=None,
           help="File name (minus extension) to save output too")
@cl.option('--plotyy', is_flag=True, help="Show plot of the target vs."
           "prediction, otherwise just save")
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
@cl.argument('targetsfile', type=cl.Path(exists=True))
@cl.argument('models', type=cl.Path(exists=True))
def main(targetsfile, files, models, plotyy, outfile):
    """
    Run cross-validation metrics on a model prediction.

    The following metrics are evaluated:
    - R-square
    - Explained variance
    - Standardised Mean Squared Error
    - Lin's concordance correlation coefficient
    - Mean Gaussian negative log likelihood (for probabilistic predictions)
    - Standardised mean Gaussian negative log likelihood (for probabilistic
      predictions)
    """

    # This runs on the root node only
    if mpiops.chunk_index != 0:
        return

    # Get all of the input data
    X, y, cv_indices = load_training_data(files, targetsfile)

    # Extract the models
    with open(models, 'rb') as f:
        models = pickle.load(f)
        cv_models = models['cross_validation']

    # Run the cross validation
    scores, y_true, y_pred = pipeline.validate(X, y, cv_models, cv_indices)

    # Log then output the scores
    for m, score in scores.items():
        log.info("{} score = {}".format(m, score))
    if outfile is not None:
        with open(outfile + ".json", 'w') as f:
            json.dump(scores, f, sort_keys=True, indent=4)

    # Make a figure if necessary
    if plotyy and (outfile is not None):
        y_y_plot(y_true, y_pred,
                 title='True vs. predicted target values.',
                 y_label='True targets',
                 y_exp_label='Predicted targets',
                 outfile=outfile,
                 display=True)
