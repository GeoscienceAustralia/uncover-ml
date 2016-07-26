"""
A demo script that ties some of the command line utilities together in a
pipeline for learning and validating models.
"""

import importlib.machinery
import logging
import json
from os import path, mkdir, listdir
from glob import glob
from mpi4py import MPI
import sys

from click import Context

from uncoverml.scripts.maketargets import main as maketargets
from uncoverml.scripts.extractfeats import main as extractfeats
from uncoverml.scripts.composefeats import main as composefeats
from uncoverml.scripts.learnmodel import main as learnmodel
from uncoverml.scripts.predict import main as predict
from uncoverml.scripts.validatemodel import main as validatemodel

# Logging
log = logging.getLogger(__name__)


# NOTE: Do not change the following unless you know what you are doing
def run_pipeline(config):

    # MPI globals
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Make processed dir if it does not exist
    if not path.exists(config.proc_dir) and rank == 0:
        mkdir(config.proc_dir)
        log.info("Made processed dir")
    comm.barrier()

    # Make sure the directory is empty if it does exist
    if listdir(config.proc_dir):
        log.fatal("Output directory must be empty!")
        sys.exit(-1)

    # Make pointspec and hdf5 for targets
    if rank == 0:
        ctx = Context(maketargets)
        ctx.forward(maketargets,
                    shapefile=path.join(config.data_dir, config.target_file),
                    fieldname=config.target_var,
                    folds=config.folds,
                    outfile=config.target_hdf,
                    seed=config.crossval_seed
                    )
    comm.barrier()

    # Extract feats for training
    tifs = glob(path.join(config.data_dir, "*.tif"))
    if len(tifs) == 0:
        log.fatal("No geotiffs found in {}!".format(config.data_dir))
        sys.exit(-1)

    # Generic extract feats command
    ctx = Context(extractfeats)

    # Find all of the tifs and extract features
    for tif in tifs:
        name = path.splitext(path.basename(tif))[0]
        log.info("Processing {}.".format(path.basename(tif)))
        ctx.forward(extractfeats,
                    geotiff=tif,
                    name=name,
                    outputdir=config.proc_dir,
                    targets=config.target_hdf,
                    onehot=config.onehot,
                    patchsize=config.patchsize
                    )
        comm.barrier()

    efiles = [f for f in glob(path.join(config.proc_dir, "*.part*.hdf5"))
              if not (path.basename(f).startswith(config.compos_file)
                      or path.basename(f).startswith(config.predict_file))]

    # Compose individual image features into single feature vector
    log.info("Composing features...")
    ctx = Context(composefeats)
    ctx.forward(composefeats,
                featurename=config.compos_file,
                outputdir=config.proc_dir,
                impute=config.impute,
                centre=config.standardise or config.whiten,
                standardise=config.standardise,
                whiten=config.whiten,
                featurefraction=config.pca_frac,
                files=efiles
                )
    comm.barrier()

    feat_files = glob(path.join(config.proc_dir,
                                config.compos_file + ".part*.hdf5"))

    for alg, args in config.algdict.items():

        # Train the model
        log.info("Training model {}.".format(alg))
        ctx = Context(learnmodel)
        ctx.forward(learnmodel,
                    outputdir=config.proc_dir,
                    crossvalidate=True,
                    algorithm=alg,
                    algopts=json.dumps(args),
                    targets=config.target_hdf,
                    files=feat_files
                    )
        comm.barrier()

        # Test the model
        log.info("Predicting targets for {}.".format(alg))
        alg_file = path.join(config.proc_dir, "{}.pk".format(alg))

        ctx = Context(predict)
        ctx.forward(predict,
                    outputdir=config.proc_dir,
                    predictname=config.predict_file + "_" + alg,
                    model=alg_file,
                    files=feat_files
                    )
        comm.barrier()

        pred_files = glob(path.join(config.proc_dir,
                                    config.predict_file + "_" + alg +
                                    ".part*.hdf5"))

        # Report score
        log.info("Validating {}.".format(alg))
        ctx = Context(validatemodel)
        ctx.forward(validatemodel,
                    outfile=path.join(config.proc_dir,
                                      config.valoutput + "_" + alg),
                    model=alg_file,
                    files=feat_files,
                    targets=config.target_hdf,

                    plotyy=False
                    )
        comm.barrier()

    log.info("Finished!")


def main():
    if len(sys.argv) != 2:
        print("Usage: learningpipeline <configfile>")
        sys.exit(-1)
    logging.basicConfig(level=logging.INFO)
    config_filename = sys.argv[1]
    config = importlib.machinery.SourceFileLoader(
        'config', config_filename).load_module()
    run_pipeline(config)

if __name__ == "__main__":
    main()
