#! /usr/bin/env python
"""
A demo script that ties some of the command line utilities together in a
pipeline for learning and validating models.
"""

import logging
import json
from os import path, mkdir
from glob import glob

from click import Context

from uncoverml import ipympi
from uncoverml.scripts.maketargets import main as maketargets
from uncoverml.scripts.cvindexer import main as cvindexer
from uncoverml.scripts.extractfeats import main as extractfeats
from uncoverml.scripts.composefeats import main as composefeats
from uncoverml.scripts.learnmodel import main as learnmodel
from uncoverml.scripts.predict import main as predict
from uncoverml.scripts.validatemodel import main as validatemodel

# from runcommands import try_run, try_run_checkfile, PipeLineFailure
from runcommands import PipeLineFailure

# logging.basicConfig(level=logging.INFO)


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
# data_dir = path.join(path.expanduser("~"), "data/GA-depth")
# data_dir = "/short/ge3/jrw547/Murray_datasets"

# Location of processed files (features, predictions etc)
proc_dir = path.join(data_dir, "processed")
# proc_dir = "/short/ge3/dms599/Murray_processed"


#
# Target Settings
#

# Shape file with target variable info
target_file = "geochem_sites.shp"
# target_file = "drillhole_confid_3.shp"
# target_file = "Targets_V8.shp"

# Target variable name (in shape file)
target_var = "Na_ppm_i_1"  # "Cr_ppm_i_1"
# target_var = "depth"

# Where to save processed targets
target_hdf = path.join(proc_dir, "{}_{}.hdf5"
                       .format(path.splitext(target_file)[0], target_var))

# Location of cross val index file. NOTE: see cvindexer tool to make these
cv_file_name = "soilcrossvalindices.hdf5"
# cv_file_name = "drillhole_xvalindices.hdf5"
cv_file = path.join(proc_dir, cv_file_name)
folds = 5


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
whiten = False  # whiten all of the extracted features?

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
    # "approxgp": {'kern': 'rbf', 'lenscale': [100.] * 87, 'nbases': 50},
    # "approxgp": {'kern': 'rbf', 'lenscale': 100., 'nbases': 50},

    # Support vector machine (regressor)
    # "svr": {'gamma': 1. / 300, 'epsilon': 0.05},
    # "svr": {},

    # Random forest regressor
    "randomforest": {'n_estimators': 500},

    # ARD Linear regression
    # "ardregression": {},

    # Kernel ridge regression
    # 'kernelridge': {'kernel': 'rbf'},

    # Decision tree regressor
    # 'decisiontree': {},

    # Extra tree regressor
    # 'extratree': {},
}

# Prediction file names (prefix)
predict_file = "prediction_file"

# Output suffix files for validation metrics
valoutput = "validation"

# Logging
log = logging.getLogger(__name__)


# NOTE: Do not change the following unless you know what you are doing
def run_pipeline():

    log.info("zzzzz")
    import time
    time.sleep(20)

    # Make processed dir if it does not exist
    if not path.exists(proc_dir):
        mkdir(proc_dir)
        log.info("Made processed dir")

    # Make pointspec and hdf5 for targets
    if not path.exists(target_hdf):
        ctx = Context(maketargets)
        ctx.forward(maketargets,
                    shapefile=path.join(data_dir, target_file),
                    fieldname=target_var,
                    outfile=target_hdf
                    )

    # Extract feats for training
    # tifs = glob(path.join(data_dir, "*.tif"))
    tifs = [path.join(data_dir, 'modis10_te.tif'),
            path.join(data_dir, 'modis7_te.tif')]
    if len(tifs) == 0:
        raise PipeLineFailure("No geotiffs found in {}!".format(data_dir))

    # Generic extract feats command
    ctx = Context(extractfeats)

    # Find all of the tifs and extract features
    for tif in tifs:
        name = path.splitext(path.basename(tif))[0]
        hdffeats = glob(path.join(proc_dir, name + '*.hdf5'))
        if len(hdffeats) == 0:
            log.info("Processing {}.".format(path.basename(tif)))
            ctx.forward(extractfeats,
                        geotiff=tif,
                        name=name,
                        outputdir=proc_dir,
                        targets=target_hdf,
                        onehot=onehot,
                        patchsize=patchsize
                        )

    efiles = [f for f in glob(path.join(proc_dir, "*.part*.hdf5"))
              if not (path.basename(f).startswith(compos_file)
                      or path.basename(f).startswith(predict_file))]

    # Make a crossval file if it doesn't exist
    if not path.exists(cv_file):
        ctx = Context(cvindexer)
        ctx.forward(cvindexer,
                    targetfile=target_hdf,
                    outfile=cv_file,
                    folds=folds
                    )

    # Compose individual image features into single feature vector
    log.info("Composing features...")
    ctx = Context(composefeats)
    ctx.forward(composefeats,
                featurename=compos_file,
                outputdir=proc_dir,
                impute=impute,
                centre=standardise or whiten,
                standardise=standardise,
                whiten=whiten,
                featurefraction=pca_frac,
                files=efiles
                )

    feat_files = glob(path.join(proc_dir, compos_file + ".part*.hdf5"))

    for alg, args in algdict.items():

        # Train the model
        # cmd = ["learnmodel", "--outputdir", proc_dir, "--cvindex", cv_file,
        #        "0", "--verbosity", "INFO", "--algorithm", alg,
        #        "--algopts", json.dumps(args)] \
        #     + feat_files + [target_hdf]

        # try_run(cmd)

        log.info("Training model {}.".format(alg))
        ctx = Context(learnmodel)
        ctx.forward(learnmodel,
                    outputdir=proc_dir,
                    cvindex=(cv_file, 0),
                    algorithm=alg,
                    algopts=json.dumps(args),
                    targets=target_hdf,
                    files=feat_files
                    )

        # Test the model
        # alg_file = path.join(proc_dir, "{}.pk".format(alg))
        # cmd = ["predict", "--outputdir", proc_dir, "--predictname",
        #        predict_file + "_" + alg, alg_file] + feat_files

        # try_run(cmd)

        log.info("Predicting targets for {}.".format(alg))
        alg_file = path.join(proc_dir, "{}.pk".format(alg))
        # import IPython; IPython.embed()

        ctx = Context(predict)
        ctx.forward(predict,
                    outputdir=proc_dir,
                    predictname=predict_file + "_" + alg,
                    model=alg_file,
                    files=feat_files
                    )

        pred_files = glob(path.join(proc_dir, predict_file + "_" + alg +
                                    ".part*.hdf5"))

        # Report score
        # cmd = ['validatemodel', '--outfile',
        #        path.join(proc_dir, valoutput + "_" + alg), cv_file, "0",
        #        target_hdf] + pred_files

        # try_run(cmd)

        log.info("Validating {}.".format(alg))
        ctx = Context(validatemodel)
        ctx.forward(validatemodel,
                    outfile=path.join(proc_dir, valoutput + "_" + alg),
                    cvindex=(cv_file, 0),
                    targets=target_hdf,
                    prediction_files=pred_files
                    )

    log.info("Finished!")


def main():

    logging.basicConfig(level=logging.INFO)
    # c = ipympi.run_ipcontroller()
    # e = [ipympi.run_ipengine() for i in range(4)]
    # ipympi.waitfor_n_engines(n=4)

    # run_pipeline()
    # c.terminate()
    # for i in e:
    #     i.terminate()

    ipympi.call_with_ipympi(run_pipeline)

if __name__ == "__main__":
    main()
