#! /usr/bin/env python3
import os
import logging
import numpy as np
import tables
import matplotlib.pyplot as pl
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor as RFR

from revrand import regression as reg
# from revrand import glm as reg
# from revrand.likelihoods import Gaussian
from revrand.basis_functions import LinearBasis, RandomRBF_ARD, RandomRBF
from yavanna.unsupervised.transforms import whiten, whiten_apply, standardise,\
    standardise_apply

from uncoverml.validation import rsquare


def main():

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Settings
    truncate = 20
    # label = 'Na_ppm_i_1'
    label = 'Cr_ppm_i_1'
    nfeatures = 200

    tr_folds = [1, 2, 3, 4]
    ts_folds = [0]

    remove_bands = [1, 2, 6]

    data_dir = "/home/dsteinberg/Code/uncover-ml/GA/"
    xval_dir = "/home/dsteinberg/data/GA-cover/"

    # Load data
    y_all = np.load(os.path.join(data_dir, "y.npy"))
    X = np.load(os.path.join(data_dir, "X.npy"))
    label_names = np.load(os.path.join(data_dir, "label_names.npy"))
    label_coords = np.load(os.path.join(data_dir, "label_coords.npy"))
    # x_bands = np.load(os.path.join(data_dir, "x_bands.npy"))
    # lons = np.load("lons.npy")
    # lats = np.load("lats.npy")
    # Xs = np.load("Xstar.npy")
    # cube = np.load("cube.npy")

    # bandinds = {v: k for k, v in enumerate(x_bands)}

    # load cross-val indices
    with tables.open_file(os.path.join(xval_dir, "soilcrossvalindices.hdf5"),
                          'r') as f:
        foldinds = np.array([i for i in f.root.FoldIndices])
        foldlons = np.array([l for l in f.root.Longitude])
        foldlats = np.array([l for l in f.root.Latitude])

    # Reconsile order of data
    assert(all(label_coords[:, 0] == foldlons) and
           all(label_coords[:, 1] == foldlats))

    # Get specific targets
    labinds = {v: k for k, v in enumerate(label_names)}
    y = y_all[:, labinds[label]]

    # Remove bands from feature data
    bandmask = np.ones(X.shape[1], dtype=bool)
    bandmask[remove_bands] = False
    X = X[:, bandmask]

    # Filter out nans
    nanmask = np.sum(np.isnan(X), axis=1) > 0
    X = X[~nanmask]
    y = y[~nanmask]
    y -= y.mean()

    # Whiten data
    Xs, Xs_mean, Xs_std = standardise(X)
    Xw = Xs
    # Xw, U, l, Xw_mean = whiten(Xs, reducedims=truncate)
    # X = whiten_apply(standardise_apply(Xf, Xmean, Xstd), U, l, Xs_w_mean)

    # Create training and testing sets
    tr_inds = np.zeros_like(foldinds, dtype=bool)
    ts_inds = np.zeros_like(foldinds, dtype=bool)

    for f in tr_folds:
        tr_inds[foldinds == f] = True
    for f in ts_folds:
        ts_inds[foldinds == f] = True

    # Filter out nans
    tr_inds = tr_inds[~nanmask]
    ts_inds = ts_inds[~nanmask]

    Xtr, ytr = Xw[tr_inds, :], y[tr_inds]
    Xts, yts = Xw[ts_inds, :], y[ts_inds]

    # Remove Lat-Lons and other layers from data
    # bandmask = np.ones(X.shape[1], dtype=bool)
    # bandmask[1] = False  # Lat/lon
    # bandmask[2] = False  # Lat/lon
    # bandmask[6] = False  # A lot of NaNs
    # bandmask[13] = False  # Super low res
    # bandmask[33] = False  # Super outlier
    # bandmask[43] = False  # Super outlier
    # bandmask[44] = False  # Super outlier
    # bandmask[45] = False  # Super outlier
    # bandmask[46] = False  # Super outlier
    # bandmask[47] = False  # Super outlier
    # bandmask[48] = False  # Super outlier and noisy

    # Regression
    # lenscale = 1.0
    # D = Xw.shape[1]
    # basis = LinearBasis(onescol=True)
    # hypers = []
    # basis = LinearBasis(onescol=True) + RandomRBF(Xdim=D, nbases=nfeatures)
    # hypers = [lenscale]
    # basis = RandomRBF_ARD(Xdim=D, nbases=nfeatures)
    # hypers = np.ones(D) * lenscale
    # basis = RandomRBF(Xdim=D, nbases=nfeatures)
    # hypers = [lenscale]
    # params = reg.learn(Xtr, ytr, basis, hypers, verbose=True, ftol=1e-6)
    # params = reg.learn_sgd(Xtr, ytr, basis, hypers, verbose=True)
    # Ey, Vf, Vy = reg.predict(Xts, basis, *params)

    # Random Forest
    rfr = RFR()
    rfr.fit(Xtr, ytr)
    Ey = rfr.predict(Xts)

    # SVM
    # svr = svm.SVR()
    # svr.fit(Xtr, ytr)
    # Ey = svr.predict(Xts)

    R2 = rsquare(Ey, yts)
    log.info("Regression r-square: {}".format(R2))

    pl.plot(Ey, yts, 'x')
    pl.title("R^2 = {}".format(R2))
    pl.ylabel("E[y]")
    pl.xlabel("y")
    pl.show()

    # import IPython; IPython.embed(); exit()


if __name__ == "__main__":
    main()
