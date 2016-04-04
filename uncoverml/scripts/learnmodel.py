"""
Learn the Parameters of a machine learning model

.. program-output:: learnmodel --help
"""

import logging
import sys
import os.path
import tables
import click as cl
import numpy as np

import uncoverml.defaults as df
from uncoverml import geoio, models

log = logging.getLogger(__name__)


@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('alg_opts', type=str, nargs=-1)
@cl.argument('algorithm', type=str, default='bayesreg')
@cl.argument('features', type=cl.Path(exists=True), nargs=-1)
@cl.argument('targets', type=cl.Path(exists=True), nargs=-1)
def main(targets, files, algorithm, algopts, outputdir, quiet):

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

    # Train the model

    # Pickle the model
