"""
Learn the Parameters of a machine learning model

.. program-output:: learnmodel --help
"""

import logging
import os.path
import pickle
import sys
import json

import numpy as np
import click as cl
import click_log as cl_log

from uncoverml import geoio
from uncoverml import mpiops, pipeline
from uncoverml.models import modelmaps


log = logging.getLogger(__name__)


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
@cl.argument('targetsfile', type=cl.Path(exists=True))
def main(targetsfile, files, algorithm, algopts, outputdir, crossvalidate):
    """
    Learn the Parameters of a machine learning model.
    """

    score_outfile = os.path.join(outputdir, "{}.json".format(algorithm))
    model_outfile = os.path.join(outputdir, "{}.pk".format(algorithm))

    # build full filenames
    full_filenames = [os.path.abspath(f) for f in files]
    log.debug("Input files: {}".format(full_filenames))

    # verify the files are all present
    files_ok = geoio.file_indices_okay(full_filenames)
    if not files_ok:
        log.fatal("Input file indices invalid!")
        sys.exit(-1)

    # build the images
    filename_dict = geoio.files_by_chunk(full_filenames)

    # Parse algorithm
    if algorithm not in modelmaps:
        log.fatal("Invalid algorthm specified")
        sys.exit(-1)

    # Parse all algorithm options
    if algopts is not None:
        args = json.loads(algopts)
    else:
        args = {}

    # Load targets file
    targets = geoio.load_targets(targetsfile)

    # Read ALL the features in here, and learn on a single machine
    X_node = geoio.load_and_cat(filename_dict[mpiops.chunk_index])
    X_list = mpiops.comm.allgather(X_node)
    X = np.ma.vstack(X_list)

    model, scores, Ys, EYs = pipeline.learn_model(X,
                                                  targets,
                                                  algorithm,
                                                  crossvalidate,
                                                  args)

    if mpiops.chunk_index == 0:
        geoio.export_scores(scores, Ys, EYs, score_outfile)

        # Pickle and store the models
        with open(model_outfile, 'wb') as f:
            pickle.dump(model, f)
