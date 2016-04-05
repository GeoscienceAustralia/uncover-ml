"""
Learn the Parameters of a machine learning model

.. program-output:: learnmodel --help
"""

import logging
import sys
import os.path
import tables
import pickle
import click as cl
import numpy as np

import uncoverml.defaults as df
from uncoverml import geoio, models

log = logging.getLogger(__name__)


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--cvindex', type=(cl.Path(exists=True), int), default=None,
           help="Optional cross validation index file and index to hold out.")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--algopts', type=str, default=None)
@cl.option('--algorithm', type=str, default='bayesreg')
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
@cl.argument('targets', type=cl.Path(exists=True))
def main(targets, files, algorithm, algopts, outputdir, cvindex, quiet):

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

    # Parse algorithm
    if algorithm not in models.modelmaps:
        log.fatal("Invalid algorthm specified")
        sys.exit(-1)

    # Parse all algorithm options
    # TODO: what format do we accept? JSON?

    # Load targets file
    with tables.open_file(targets, mode='r') as f:
        y = f.root.targets.read()

    # Read ALL the features in here, and learn on a single machine
    # FIXME?
    feats = []
    for f in full_filenames:
        with tables.open_file(f, mode='r') as tab:
            feats.append(tab.root.features.read())

    X = np.vstack(feats)

    # Optionally subset the data for cross validation
    if cvindex:
        with tables.open_file(cvindex[0], mode='r') as f:
            cv_ind = f.root.FoldIndices.read().flatten()

        y = y[cv_ind != cvindex[1]]
        X = X[cv_ind != cvindex[1]]

    # Train the model
    mod = models.modelmaps[algorithm]()  # TODO: input params
    mod.fit(X, y)

    # Pickle the model
    outfile = os.path.join(outputdir, "{}.pk".format(algorithm))
    with open(outfile, 'wb') as f:
        pickle.dump(mod, f)
