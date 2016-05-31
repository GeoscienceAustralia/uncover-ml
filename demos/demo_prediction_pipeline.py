#! /usr/bin/env python
"""
A demo script that ties some of the command line utilities together in a
pipeline for prediction.
"""

import sys
import logging
from os import path, mkdir
from glob import glob

from runcommands import try_run, try_run_checkfile, PipeLineFailure

log = logging.getLogger(__name__)


# NOTE: INSTRUCTIONS ----------------------------------------------------------
#   1) Make sure you have an ipcluster working running, i.e.
#       $ ipcluster start --n=4
#   2) Make sure you have all of the data and cross-val file in the directory
#      structure specified below (or change it to suit your purposes)
#   3) Make sure you have learned a model by running
#       $ ./demo_learning_pipeline.py
#   4) run this script, e.g.
#       $ ./demo_prediction_pipeline.py
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

# Location of processed file (features, predictions etc)
proc_dir = path.join(data_dir, "processed")

# Location of the prediction output from this script
pred_dir = path.join(proc_dir, "prediction")

# Composite feature names (prefixes)
compos_file = "composite"


#
# Prediction settings
#

# Number of jobs (must be >= number of workers)
nchunks = 12

# Name of the prediction algorithm
# algorithm = 'svr'
# algorithm = 'bayesreg'
algorithm = 'approxgp'
# algorithm = 'randomforest'

# Prediction file names (prefix)
predict_file = "prediction_file"


#
# Visualisation/Geotiff settings
#

# Name of the prediction output tif
gtiffname = "prediction_image"
gtiffname_ent = "entropy_reduction_image"

# Make the image RGB?
makergbtif = True


# NOTE: Do not change the following unless you know what you are doing
def main():

    logging.basicConfig(level=logging.INFO)

    if not path.exists(proc_dir):
        log.fatal("Please run demo_learning_pipline.py first!")
        sys.exit(-1)

    # Make processed dir if it does not exist
    if not path.exists(pred_dir):
        mkdir(pred_dir)
        log.info("Made prediction dir")

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

    # Get suffix of end processed file for checking completion of commands
    endsuf = ".part" + str(nchunks - 1) + ".hdf5"

    # Now Extact features from each tif
    cmd = ["extractfeats", None, None, "--outputdir", pred_dir, "--chunks",
           str(nchunks), "--settings", None]
    for tif, setting in zip(tifs, settings):
        msg = "Processing {}.".format(path.basename(tif))
        name = path.splitext(path.basename(tif))[0]
        cmd[1], cmd[2], cmd[-1] = name, tif, setting
        ffile = path.join(pred_dir, name + endsuf)
        try_run_checkfile(cmd, ffile, msg)

    # Compose individual image features into single feature vector
    compos_settings = path.join(proc_dir, compos_file + "_settings.bin")
    if not path.exists(compos_settings):
        log.fatal("Settings file for composite features does not exist!")
        sys.exit(-1)

    efiles = [f for f in glob(path.join(pred_dir, "*.hdf5"))
              if not path.basename(f).startswith(compos_file)]
    cmd = ["composefeats", "--outputdir", pred_dir, "--settings",
           compos_settings, compos_file] + efiles

    ffile = path.join(pred_dir, compos_file + endsuf)
    try_run_checkfile(cmd, ffile, "Composing features...")

    # Now predict on the composite features!
    alg_file = path.join(proc_dir, "{}.pk".format(algorithm))
    if not path.exists(alg_file):
        log.fatal("Learned algorithm file {} missing!".format(alg_file))
        sys.exit(-1)
    cfiles = glob(path.join(pred_dir, compos_file + "*.hdf5"))

    cmd = ["predict", "--outputdir", pred_dir,
           "--predictname", predict_file]
    cmd += [alg_file] + cfiles

    pfile = path.join(pred_dir, predict_file + endsuf)
    try_run_checkfile(cmd, pfile, "Predicting targets...")

    # General export command
    cmd = ["exportgeotiff", gtiffname, "--outputdir", pred_dir]
    if makergbtif:
        cmd += ["--rgb"]

    # Output a Geotiff of the predictions
    pfiles = glob(path.join(pred_dir, predict_file + "*.hdf5"))
    cmdp = cmd + pfiles
    try_run(cmdp)


if __name__ == "__main__":
    main()
