"""
Learn the Parameters of a machine learning model

.. program-output:: learnmodel --help
"""

import logging
import os.path
import pickle

import click as cl
import click_log as cl_log

from uncoverml import mpiops, pipeline
from uncoverml.geoio import load_training_data
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

    # This runs on the root node only
    if mpiops.chunk_index != 0:
        return

    # Get all of the required data and learn the required models
    X, y, cv_indices = load_training_data(files, targetsfile)
    models = pipeline.learn_model(X, y,
                                  algorithm,
                                  cv_indices if crossvalidate else None,
                                  algopts)

    # Pickle and store the models
    outfile = os.path.join(outputdir, "{}.pk".format(algorithm))
    with open(outfile, 'wb') as f:
        pickle.dump(models, f)
