"""
Learn the Parameters of a machine learning model

.. program-output:: learnmodel --help
"""

import logging
import sys
import os.path
import pickle
import json
import click as cl
import click_log as cl_log
import numpy as np
from mpi4py import MPI

from uncoverml import geoio
# from uncoverml.validation import input_cvindex, input_targets
from uncoverml.models import modelmaps, apply_multiple_masked


log = logging.getLogger(__name__)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--cvindex', type=int, default=None,
           help="Optional cross validation index to hold out.")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--algopts', type=str, default=None, help="JSON string of optional "
           "parameters to pass to the learning algorithm.")
@cl.option('--algorithm', type=cl.Choice(list(modelmaps.keys())),
           default='bayesreg', help="algorithm to learn.")
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
@cl.argument('targets', type=cl.Path(exists=True))
def main(targets, files, algorithm, algopts, outputdir, cvindex):
    """
    Learn the Parameters of a machine learning model.
    """

    # MPI globals
    comm = MPI.COMM_WORLD
    chunk_index = comm.Get_rank()
    # This runs on the root node only
    if chunk_index != 0:
        return

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
    nchunks = len(filename_dict)

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
    ydict = geoio.points_from_hdf(targets, ['targets_sorted',
                                            'FoldIndices_sorted'])
    y = ydict['targets_sorted']

    # Read ALL the features in here, and learn on a single machine
    data_vectors = [geoio.load_and_cat(filename_dict[i])
                    for i in range(nchunks)]
    # Remove the missing data
    data_vectors = [x for x in data_vectors if x is not None]
    X = np.ma.concatenate(data_vectors, axis=0)

    # Optionally subset the data for cross validation
    print("X shape: {}".format(X.shape))
    print("y shape: {}".format(y.shape))
    if cvindex is not None:
        cv_ind = ydict['FoldIndices_sorted']
        print("cv_ind shape: {}".format(cv_ind.shape))
        y = y[cv_ind != cvindex]
        X = X[cv_ind != cvindex]

    # Train the model
    mod = modelmaps[algorithm](**args)
    apply_multiple_masked(mod.fit, (X, y))

    # Pickle the model
    outfile = os.path.join(outputdir, "{}.pk".format(algorithm))
    with open(outfile, 'wb') as f:
        pickle.dump(mod, f)
