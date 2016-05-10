#! /usr/bin/env python
"""
A demo script that ties some of the command line utilities together in a
pipeline for learning and validating models.
"""

import logging
import json
from os import path, mkdir
from glob import glob

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
# data_dir = path.join(path.expanduser("~"), "data/GA-cover")
data_dir = path.join(path.expanduser("~"), "data/GA-depth")

# Location of processed files (features, predictions etc)
proc_dir = path.join(data_dir, "processed")


#
# Target Settings
#

# Shape file with target variable info
# target_file = "geochem_sites.shp"
target_file = "drillhole_confid_3.shp"

# Target variable name (in shape file)
# target_var = "Na_ppm_i_1"  # "Cr_ppm_i_1"
target_var = "depth"

# Where to save processed targets
target_hdf = path.join(proc_dir, "{}_{}.hdf5"
                       .format(path.splitext(target_file)[0], target_var))

# Location of cross val index file. NOTE: see cvindexer tool to make these
# cv_file_name = "soilcrossvalindices.hdf5"
cv_file_name = "drillhole_xvalindices.hdf5"
cv_file = path.join(data_dir, cv_file_name)


#
# Feature settings
#

# Automatically detect integer-valued files and use one-hot encoding?
onehot = False  # NOTE: if you change this, make sure you delete all old feats

# Patch size to extract around targets (0 = 1x1 pixel, 1 = 3x3 pixels etc)
patchsize = 0  # NOTE: if you change this, make sure you delete all old feats

# Impute missing values?
impute = True

# Starndardise each input dimension? (0 mean, 1 std)
standardise = True  # standardise all of the extracted features?

# Whiten all inputs?
whiten = True  # whiten all of the extracted features?

# Fraction of dimensions to keep *if* whitening
pca_frac = 0.5

# Composite feature names (prefixes)
compos_file = "composite"


#
# Algorithm settings
#

# Iterate through this dictionary of algorithm name and arguments:
algdict = {

    # Bayesian linear regression
    # "bayesreg": {},

    # Approximate Gaussian process, for large scale data
    "approxgp": {'kern': 'matern32', 'lenscale': [100.] * 43, 'nbases': 200}

    # Support vector machine (regressor)
    # "svr": {'gamma': 1. / 90, 'epsilon': 0.05}

    # Random forest regressor
    # "randomforest": {'n_estimators': 20}

    # ARD Linear regression
    # "ardregression": {},

    # Kernel ridge regression
    # 'kernelridge': {'kernel': 'rbf'},

    # Decision tree regressor
    # 'deciciontree': {},

    # Extra tree regressor
    # 'extratree': {},
}

# Prediction file names (prefix)
predict_file = "prediction_file"

# Output suffix files for validation metrics
valoutput = "validation"


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

    for alg, args in algdict.items():

        # Train the model
        cmd = ["learnmodel", "--outputdir", proc_dir, "--cvindex", cv_file,
               "0", "--algorithm", alg, "--algopts", json.dumps(args),
               feat_file, target_hdf]

        log.info("Training model {}.".format(alg))
        try_run(cmd)

        # Test the model
        alg_file = path.join(proc_dir, "{}.pk".format(alg))
        cmd = ["predict", "--outputdir", proc_dir, "--cvindex", cv_file, "0",
               "--predictname", predict_file + "_" + alg, alg_file, feat_file]

        log.info("Predicting targets for {}.".format(alg))
        try_run(cmd)

        # Report score
        cmd = ['validatemodel', '--outfile',
               path.join(proc_dir, valoutput + "_" + alg), cv_file, "0",
               target_hdf,
               path.join(proc_dir, predict_file + "_" + alg + ".part0.hdf5")]

        log.info("Validating {}.".format(alg))
        try_run(cmd)

    log.info("Finished!")


if __name__ == "__main__":
    main()
