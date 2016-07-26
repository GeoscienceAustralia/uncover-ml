#! /usr/bin/env python
"""
A demo script that ties some of the command line utilities together in a
pipeline for prediction.
"""

import importlib.machinery
import sys
import logging
from os import path, mkdir, listdir
from glob import glob
from mpi4py import MPI

from click import Context

from uncoverml.scripts.extractfeats import main as extractfeats
from uncoverml.scripts.composefeats import main as composefeats
from uncoverml.scripts.predict import main as predict
from uncoverml.scripts.exportgeotiff import main as exportgeotiff

log = logging.getLogger(__name__)


def run_pipeline(config):

    # MPI globals
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if not path.exists(config.proc_dir):
        log.fatal("Please run demo_learning_pipline.py first!")
        sys.exit(-1)

    # Make processed dir if it does not exist
    if not path.exists(config.pred_dir) and rank == 0:
        mkdir(config.pred_dir)
        log.info("Made prediction dir")
    comm.barrier()

    # Make sure prediction dir is empty
    if listdir(config.pred_dir):
        log.fatal("Prediction directory must be empty!")
        sys.exit(-1)

    # Make sure we have an extractfeats settings file for each tif
    tifs = glob(path.join(config.data_dir, "*.tif"))
    if len(tifs) == 0:
        log.fatal("No geotiffs found in {}!".format(config.data_dir))
        sys.exit(-1)

    settings = []
    for tif in tifs:
        setting = path.join(config.proc_dir, 
                            path.splitext(path.basename(tif))[0] +
                            "_settings.bin")
        if not path.exists(setting):
            log.fatal("Setting file {} does not exist!".format(setting))
            sys.exit(-1)
        settings.append(setting)

    # Now extact features from each tif
    ctx = Context(extractfeats)

    # Find all of the tifs and extract features
    for tif, setting in zip(tifs, settings):
        name = path.splitext(path.basename(tif))[0]
        log.info("Processing {}.".format(path.basename(tif)))
        ctx.forward(extractfeats,
                    geotiff=tif,
                    name=name,
                    outputdir=config.pred_dir,
                    settings=setting
                    )
        comm.barrier()

    # Compose individual image features into single feature vector
    compos_settings = path.join(config.proc_dir, 
                                config.compos_file + "_settings.bin")
    if not path.exists(compos_settings):
        log.fatal("Settings file for composite features does not exist!")
        sys.exit(-1)

    efiles = [f for f in glob(path.join(config.pred_dir, "*.part*.hdf5"))
              if not (path.basename(f).startswith(config.compos_file) or
                      path.basename(f).startswith(config.predict_file))]

    log.info("Composing features...")
    ctx = Context(composefeats)
    ctx.forward(composefeats,
                name=config.compos_file,
                outputdir=config.pred_dir,
                files=efiles,
                settings=compos_settings
                )
    comm.barrier()

    cfiles = glob(path.join(config.pred_dir, config.compos_file + "*.hdf5"))

    # Now predict on the composite features!
    alg_file = path.join(config.proc_dir, "{}.pk".format(config.algorithm))
    if not path.exists(alg_file):
        log.fatal("Learned algorithm file {} missing!".format(alg_file))
        sys.exit(-1)

    alg = path.splitext(path.basename(alg_file))[0]
    predict_alg_file = config.predict_file + '_' + alg

    log.info("Predicting targets...")
    ctx = Context(predict)
    ctx.forward(predict,
                outputdir=config.pred_dir,
                predictname=predict_alg_file,
                model=alg_file,
                quantiles=config.quantiles,
                files=cfiles
                )
    comm.barrier()

    pfiles = glob(path.join(config.pred_dir, predict_alg_file + '*.hdf5'))

    # Output a Geotiff of the predictions
    log.info("Exporting geoftiffs...")
    ctx = Context(exportgeotiff)
    ctx.forward(exportgeotiff,
                name=config.gtiffname + "_" + alg,
                outputdir=config.pred_dir,
                rgb=config.makergbtif,
                files=pfiles
                )
    comm.barrier()

    log.info("Done!")


def main():
    if len(sys.argv) != 2:
        print("Usage: predictionpipeline <configfile>")
        sys.exit(-1)
    logging.basicConfig(level=logging.INFO)
    config_filename = sys.argv[1]
    config = importlib.machinery.SourceFileLoader(
        'config', config_filename).load_module()
    run_pipeline(config)

if __name__ == "__main__":
    main()
