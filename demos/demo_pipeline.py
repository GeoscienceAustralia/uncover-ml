#! /usr/bin/env python3
"""
A demo script that ties some of the command line utilities together in a
pipeline

TODO: Replicate this with luigi or joblib
"""

import sys
import logging
from os import path, mkdir
from glob import glob
from subprocess import run, CalledProcessError

log = logging.getLogger(__name__)


# Settings
data_dir = path.join(path.expanduser("~"), "data/GA-cover")
proc_dir = path.join(data_dir, "processed")

target_file = "geochem_sites.shp"
target_var = "Na_ppm_i_1"
target_hdf = path.join(proc_dir, "{}_{}.hdf5"
                       .format(path.splitext(target_file)[0], target_var))

input_file = "inputs.npz"
feature_file = "features.npz"
whiten = True  # whiten all of the extracted features?
pca_dims = 20  # if whitening, how many PCA dimensions to keep?


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

    for tif in tifs:
        name = path.splitext(path.basename(tif))[0]
        cmd = ["extractfeats", tif, name, "--outputdir", proc_dir, "--chunks",
               "1", "--targets", target_hdf, "--standalone"]
        msg = "Processing {}.".format(path.basename(tif))
        try_run_checkfile(cmd, path.join(proc_dir, name + "_0.hdf5"), msg)

    # Compose individual image features into single feature vector
    # TODO use a scrip for this

    # Whiten the features (?)

    # Save whitening parameters to model spec

    # Divide the features into cross-val folds

    # Train the model

    # Test the model

    # Report score

    log.info("Done!")


class PipeLineFailure(Exception):
    pass


def combinefeats(tifs):

    pass


def try_run_checkfile(cmd, checkfile, premsg=None):

    if not path.exists(checkfile):
        if premsg:
            log.info(premsg)
        try_run(cmd)
        return True

    return False


def try_run(cmd):

    try:
        run(cmd, check=True)
    except CalledProcessError:
        log.info("\n--------------------\n")
        raise


if __name__ == "__main__":
    main()
