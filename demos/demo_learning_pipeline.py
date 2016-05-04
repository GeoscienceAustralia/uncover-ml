#! /usr/bin/env python
"""
A demo script that ties some of the command line utilities together in a
pipeline for learning and validating models.
"""

# TODO:
# - make sure this can run for multiple chunks
# - include the ability to iterate through algorithms

import logging
import json
from os import path, mkdir
from glob import glob

from uncoverml.models import probmodels_str
from runcommands import try_run, try_run_checkfile, PipeLineFailure

log = logging.getLogger(__name__)


# NOTE: INSTRUCTIONS ----------------------------------------------------------
#   1) Make sure you have an ipcluster working running, i.e.
#       $ ipcluster start --n=1
#   2) Make sure you have all of the data and cross-val file in the directory
#      structure specified below (or change it to suit your purposes)
#   3) run this script, e.g.
#       $ ./demo_learning_pipline.py
# -----------------------------------------------------------------------------


#
# Path Settings
#

# As this is set now, it assumes a folder structure like the following:
# ~/data/GA-cover   << location of data (all tif and shp files)
# ~/data/GA-cover/processed     << location of all output files
# ~/data/GA-cover/soilcrossvalindices.hdf5   << cross val hdf5 file from us

# please change the following paths to suit your needs

# Location of data
data_dir = path.join(path.expanduser("~"), "data/GA-cover")

# Location of processed files (features, predictions etc)
proc_dir = path.join(data_dir, "processed")


#
# Target Settings
#

# Shape file with target variable info
target_file = "geochem_sites.shp"

# Target variable name (in shape file)
target_var = "Na_ppm_i_1"  # "Cr_ppm_i_1"

# Where to save processed targets
target_hdf = path.join(proc_dir, "{}_{}.hdf5"
                       .format(path.splitext(target_file)[0], target_var))

# Location of cross val index file. NOTE: see cvindexer tool to make these
cv_file = path.join(data_dir, "soilcrossvalindices.hdf5")


#
# Feature settings
#

# Automatically detect integer-valued files and use one-hot encoding?
onehot = False  # NOTE: if you change this, make sure you delete all old feats

# Patch size to extract around targets (0 = 1x1 pixel, 1 = 3x3 pixels etc)
patchsize = 0  # NOTE: if you change this, make sure you delete all old feats

# Impute missing values?
impute = False

# Starndardise each input dimension? (0 mean, 1 std)
standardise = False  # standardise all of the extracted features?

# Whiten all inputs?
whiten = True  # whiten all of the extracted features?

# Fraction of dimensions to keep *if* whitening
pca_frac = 0.7

# Composite feature names (prefixes)
compos_file = "composite"


#
# Algorithm settings
#

# Bayesian linear regression
# algorithm = "bayesreg"
# args = {}

# Approximate Gaussian process, for large scale data
algorithm = "approxgp"
args = {'kern': 'matern32', 'lenscale': 1000., 'nbases': 200}

# Support vector machine (regressor)
# algorithm = "svr"
# args = {'gamma': 1. / 70, 'epsilon': 0.05}

# Random forest regressor
# algorithm = "randomforest"
# args = {'n_estimators': 20}

# Prediction file names (prefix)
predict_file = "prediction_file"


#
# Validation settings
#

metrics = ['r2_score', 'smse', 'lins_ccc']

# Extra settings if the model is probabilistic
if algorithm in probmodels_str:
    metrics.append('msll')


# NOTE: Do not change the following unless you know what you are doing
def main():

    logging.basicConfig(level=logging.INFO)

    # Make processed dir if it does not exist
    if not path.exists(proc_dir):
        mkdir(proc_dir)
        log.info("Made processed dir")

    # Make pointspec and hdf5 for targets
    cmd = ["maketargets", path.join(data_dir, target_file), target_var,
           "--outfile", target_hdf]

    if try_run_checkfile(cmd, target_hdf):
        log.info("Made targets")

    # Extract feats for training
    tifs = glob(path.join(data_dir, "*.tif"))
    if len(tifs) == 0:
        raise PipeLineFailure("No geotiffs found in {}!".format(data_dir))

    # Generic extract feats command
    cmd = ["extractfeats", None, None, "--outputdir", proc_dir, "--chunks",
           "1", "--patchsize", str(patchsize), "--targets", target_hdf]
    if onehot:
        cmd.append('--onehot')

    # Find all of the tifs and extract features
    ffiles = []
    for tif in tifs:
        msg = "Processing {}.".format(path.basename(tif))
        name = path.splitext(path.basename(tif))[0]
        cmd[1], cmd[2] = name, tif
        ffile = path.join(proc_dir, name + ".part0.hdf5")
        try_run_checkfile(cmd, ffile, msg)
        ffiles.append(ffile)

    # Compose individual image features into single feature vector
    cmd = ["composefeats"]
    if impute:
        cmd.append('--impute')
    if standardise:
        cmd += ['--centre', '--standardise']
    if whiten:
        cmd += ['--whiten', '--featurefraction', str(pca_frac)]
    cmd += ['--outputdir', proc_dir, compos_file] + ffiles

    feat_file = path.join(proc_dir, compos_file + ".part0.hdf5")
    try_run(cmd)

    # Train the model
    cmd = ["learnmodel", "--outputdir", proc_dir, "--cvindex", cv_file, "0",
           "--algorithm", algorithm, "--algopts", json.dumps(args), feat_file,
           target_hdf]

    log.info("Training model.")
    try_run(cmd)

    # Test the model
    alg_file = path.join(proc_dir, "{}.pk".format(algorithm))
    cmd = ["predict", "--outputdir", proc_dir, "--cvindex", cv_file, "0",
           "--predictname", predict_file, alg_file, feat_file]

    log.info("Predicting targets.")
    try_run(cmd)

    # Report score
    log.info("Validating model.")

    for i, m in enumerate(metrics):
        cmd = ["validatemodel", "--metric", m]
        if i == (len(metrics) - 1):
            cmd.append('--plotyy')
        cmd += [cv_file, "0", target_hdf,
                path.join(proc_dir, predict_file + ".part0.hdf5")]

        try_run(cmd)

    log.info("Finished!")


if __name__ == "__main__":
    main()
