#! /usr/bin/env python
"""
A demo script that ties some of the command line utilities together in a
pipeline for prediction.
"""

import sys
import logging
from os import path, mkdir
from glob import glob
from mpi4py import MPI

from click import Context

from uncoverml.scripts.extractfeats import main as extractfeats
from uncoverml.scripts.composefeats import main as composefeats
from uncoverml.scripts.predict import main as predict
from uncoverml.scripts.exportgeotiff import main as exportgeotiff

from runcommands import PipeLineFailure


log = logging.getLogger(__name__)


# NOTE: INSTRUCTIONS ----------------------------------------------------------
#   1) Make sure you have open MPI install (i.e. mpirun)
#   2) Make sure you have all of the data in the directory structure specified
#      below (or change it to suit your purposes)
#   3) Make sure you have a learned model by running demo_learning_pipeline.py
#   4) run this script using mpirun with n + 2 workers, e.g.
#       $ mpirun -n 6 demo_prediction_pipline.py
#      for 4 workers (we need two workers for coordination)
# -----------------------------------------------------------------------------

#
# Path Settings
#

# As this is set now, it assumes a folder structure like the following:
# ~/data/GA-cover   << location of data (all tif and shp files)
# ~/data/GA-cover/processed     << location of output from learning pipeline
# ~/data/GA-cover/processed/prediction     << location of prediction output

# please change the following paths to suit your needs

# Location of data
data_dir_name = "data/GA-cover"
# data_dir_name = "data/GA-depth"
data_dir = path.join(path.expanduser("~"), data_dir_name)
# data_dir = "/short/ge3/jrw547/Murray_datasets"
# data_dir = "/short/ge3/jrw547/GA-cover"

# Location of processed file (features, predictions etc)
# proc_dir = "/short/ge3/dms599/Murray_processed"
# proc_dir = "/short/ge3/dms599/GA-cover_processed"
proc_dir = path.join(data_dir, "processed")

# Location of the prediction output from this script
pred_dir = path.join(proc_dir, "prediction")

# Composite feature names (prefixes)
compos_file = "composite"


#
# Prediction settings
#

# Name of the prediction algorithm
# algorithm = 'svr'
# algorithm = 'bayesreg'
# algorithm = 'approxgp'
algorithm = 'randomforest'

# Prediction file names (prefix)
predict_file = "prediction"

# Quantiles
quantiles = 0.95


#
# Visualisation/Geotiff settings
#

# Name of the prediction output tif
gtiffname = "prediction_image"
gtiffname_ent = "entropy_reduction_image"

# Make the image RGB?
makergbtif = True


# NOTE: Do not change the following unless you know what you are doing
def run_pipeline():

    # MPI globals
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if not path.exists(proc_dir):
        log.fatal("Please run demo_learning_pipline.py first!")
        sys.exit(-1)

    # Make processed dir if it does not exist
    if not path.exists(pred_dir) and rank == 0:
        mkdir(pred_dir)
        log.info("Made prediction dir")

    comm.barrier()

    # Make sure we have an extractfeats settings file for each tif
    tifs = glob(path.join(data_dir, "*.tif"))
    if len(tifs) == 0:
        raise PipeLineFailure("No geotiffs found in {}!".format(data_dir))

    settings = []
    for tif in tifs:
        setting = path.join(proc_dir, path.splitext(path.basename(tif))[0]
                            + "_settings.bin")
        if not path.exists(setting):
            log.fatal("Setting file for {} does not exist!".format(tif))
            sys.exit(-1)
        settings.append(setting)

    # Now Extact features from each tif
    ctx = Context(extractfeats)

    # Find all of the tifs and extract features
    for tif, setting in zip(tifs, settings):
        name = path.splitext(path.basename(tif))[0]
        hdffeats = glob(path.join(pred_dir, name + '*.hdf5'))
        if len(hdffeats) == 0:
            log.info("Processing {}.".format(path.basename(tif)))
            ctx.forward(extractfeats,
                        geotiff=tif,
                        name=name,
                        outputdir=pred_dir,
                        settings=setting
                        )
            comm.barrier()

    # Compose individual image features into single feature vector
    compos_settings = path.join(proc_dir, compos_file + "_settings.bin")
    if not path.exists(compos_settings):
        log.fatal("Settings file for composite features does not exist!")
        sys.exit(-1)

    efiles = [f for f in glob(path.join(pred_dir, "*.part*.hdf5"))
              if not (path.basename(f).startswith(compos_file)
                      or path.basename(f).startswith(predict_file))]

    cfiles = glob(path.join(pred_dir, compos_file + "*.hdf5"))
    if len(cfiles) == 0:
        log.info("Composing features...")
        ctx = Context(composefeats)
        ctx.forward(composefeats,
                    featurename=compos_file,
                    outputdir=pred_dir,
                    files=efiles,
                    settings=compos_settings
                    )
        comm.barrier()

        cfiles = glob(path.join(pred_dir, compos_file + "*.hdf5"))

    # Now predict on the composite features!
    alg_file = path.join(proc_dir, "{}.pk".format(algorithm))
    if not path.exists(alg_file):
        log.fatal("Learned algorithm file {} missing!".format(alg_file))
        sys.exit(-1)

    alg = path.splitext(path.basename(alg_file))[0]
    predict_alg_file = predict_file + '_' + alg

    if rank == 0:
        print(cfiles)

    pfiles = glob(path.join(pred_dir, predict_alg_file + '*.hdf5'))
    if len(pfiles) == 0:
        log.info("Predicting targets...")
        ctx = Context(predict)
        ctx.forward(predict,
                    outputdir=pred_dir,
                    predictname=predict_alg_file,
                    model=alg_file,
                    quantiles=quantiles,
                    files=cfiles
                    )
        comm.barrier()

        pfiles = glob(path.join(pred_dir, predict_alg_file + '*.hdf5'))

    # Output a Geotiff of the predictions
    log.info("Exporting geoftiffs...")
    ctx = Context(exportgeotiff)
    ctx.forward(exportgeotiff,
                name=gtiffname + "_" + alg,
                outputdir=pred_dir,
                rgb=makergbtif,
                files=pfiles
                )
    comm.barrier()

    log.info("Done!")


def main():

    logging.basicConfig(level=logging.INFO)
    run_pipeline()

if __name__ == "__main__":
    main()
