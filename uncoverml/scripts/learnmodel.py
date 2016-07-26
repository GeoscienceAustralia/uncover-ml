"""
Learn the Parameters of a machine learning model

.. program-output:: learnmodel --help
"""


import json

import logging

import os.path

import pickle

import sys

import click as cl

import click_log as cl_log

from mpi4py import MPI

import numpy as np

from uncoverml import geoio

from uncoverml.models import apply_multiple_masked, modelmaps

log = logging.getLogger(__name__)


def exists(item):
    return item is not None


def exit(message):
    log.fatal(message)
    sys.exit(-1)


def train(model, X, y, index_mask=None):

    # Remove the rows in this fold
    X_fold = X[index_mask] if index_mask else X
    y_fold = y[index_mask] if index_mask else y

    # Train the model for this fold
    apply_multiple_masked(model.fit, (X_fold, y_fold))


def load_training_data(files, targets):

    # Build full filenames
    full_filenames = [os.path.abspath(f) for f in files]
    log.debug("Input files: {}".format(full_filenames))

    # Verify the files are all present
    files_ok = geoio.file_indices_okay(full_filenames)
    if not files_ok:
        exit("Input file indices invalid!")

    # Build the images
    filename_dict = geoio.files_by_chunk(full_filenames)
    nchunks = len(filename_dict)

    # Read ALL the features in here and remove any missing data for the X's
    data_vectors = [geoio.load_and_cat(filename_dict[i])
                    for i in range(nchunks)]
    data_vectors = list(filter(exists, data_vectors))
    X = np.ma.concatenate(data_vectors, axis=0)

    # Load the targets file to produce the y's and cross validation indices
    ydict = geoio.points_from_hdf(targets, ['targets_sorted',
                                            'FoldIndices_sorted'])
    y = ydict['targets_sorted']
    cv_indices = ydict['FoldIndices_sorted']

    return (X, y, cv_indices)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--crossvalidate', default=False,
           help="Trains K cross validation models")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--algopts', type=str, default=None, help="JSON string of optional "
           "parameters to pass to the learning algorithm.")
@cl.option('--algorithm', type=cl.Choice(list(modelmaps.keys())),
           default='bayesreg', help="algorithm to learn.")
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
@cl.argument('targets', type=cl.Path(exists=True))
def main(targets, files, algorithm, algopts, outputdir, crossvalidate):
    """ Learn the Parameters of a machine learning model. """

    # MPI globals
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # This runs on the root node only, so we learn only on a single machine
    if rank != 0:
        return

    # Get all of the required data
    X, y, cv_indices = load_training_data(files, targets)

    # Determine the required algorithm and parse it's options
    if algorithm not in modelmaps:
        exit("Invalid algorthm specified")
    args = json.loads(algopts) if algopts is not None else {}

    # Train the master model and store it
    model = modelmaps[algorithm](**args)
    train(model, X, y)
    models = dict()
    models['master'] = model

    # Train the cross validation models if necessary
    if crossvalidate:

        # Populate the validation indices
        models['cross_validation'] = []
        models['cv_indices'] = cv_indices

        # Train each model and store it
        for fold in range(max(cv_indices) + 1):

            # Train a model for each row
            remaining_rows = [cv_indices != fold]
            model = modelmaps[algorithm](**args)
            train(model, X, y, remaining_rows)

            # Store the model parameters
            models['cross_validation'].append(model)

    # Pickle and store the models
    outfile = os.path.join(outputdir, "{}.pk".format(algorithm))
    with open(outfile, 'wb') as f:
        pickle.dump(models, f)
