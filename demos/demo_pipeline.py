#! /usr/bin/env python3
"""
A demo script that ties some of the command line utilities together in a
pipeline

TODO: Replicate this with luigi or joblib
"""

import logging
import tables
import numpy as np
from os import path, mkdir
from glob import glob
from subprocess import run, CalledProcessError

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

from revrand.regression import learn, predict
from revrand.basis_functions import LinearBasis, RandomRBF, RandomRBF_ARD
from uncoverml import validation

log = logging.getLogger(__name__)


# Settings
data_dir = path.join(path.expanduser("~"), "data/GA-cover")
proc_dir = path.join(data_dir, "processed")

target_file = "geochem_sites.shp"
target_var = "Na_ppm_i_1"
target_hdf = path.join(proc_dir, "{}_{}.hdf5"
                       .format(path.splitext(target_file)[0], target_var))
cv_file = path.join(data_dir, "soilcrossvalindices.hdf5")

# input_file = "inputs.npz"
# feature_file = "features.npz"
whiten = False  # whiten all of the extracted features?
standardise = True  # standardise all of the extracted features?
pca_dims = 15  # if whitening, how many PCA dimensions to keep?


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

    ffiles = []
    for tif in tifs:
        name = path.splitext(path.basename(tif))[0]
        cmd = ["extractfeats", tif, name, "--outputdir", proc_dir, "--chunks",
               "1", "--targets", target_hdf, "--standalone"]
        msg = "Processing {}.".format(path.basename(tif))
        ffile = path.join(proc_dir, name + "_0.hdf5")
        try_run_checkfile(cmd, ffile, msg)
        ffiles.append(ffile)

    # Compose individual image features into single feature vector
    # TODO use a script for this
    feats = []
    for ffile in ffiles:
        with tables.open_file(ffile, mode='r') as f:
            feat = f.root.features.read()
            feats.append(feat)

    X = np.hstack(feats)

    # Int to one-hot TODO

    # Standardise features
    if standardise:
        log.info("Standartising the features.")
        X -= X.mean(axis=0)
        X /= X.std(axis=0)

    # Whiten the features
    if whiten:
        log.info("Whitening the features.")
        pca = PCA(n_components=pca_dims, whiten=False)
        X = pca.fit_transform(X)

    D = X.shape[1]

    # Save whitening parameters to model spec
    # TODO

    # Divide the features into cross-val folds
    with tables.open_file(cv_file, mode='r') as f:
        cv_ind = f.root.FoldIndices.read().flatten()

    Xs = X[cv_ind == 0]
    Xt = X[cv_ind != 0]

    with tables.open_file(target_hdf, mode='r') as f:
        Y = f.get_node('/' + target_var).read()

    Ys = Y[cv_ind == 0]
    Yt = Y[cv_ind != 0]

    # Train the model
    log.info("Training model.")
    basis = RandomRBF(nbases=700, Xdim=D) + LinearBasis(onescol=True)
    hypers = 10 * np.ones(1)
    params = learn(Xt, Yt, basis, hypers)
    # rfr = RandomForestRegressor()
    # rfr.fit(Xt, Yt)

    # Test the model
    log.info("Testing model.")
    # EYs = rfr.predict(Xs)
    EYs, Vfs, VYs = predict(Xs, basis, *params)

    # Report score
    Rsquare = validation.rsquare(EYs, Ys)

    log.info("Done! R-square = {}".format(Rsquare))


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
