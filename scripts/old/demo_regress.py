#! /usr/bin/env python3
import logging
import numpy as np
import matplotlib.pyplot as pl
import revrand.legacygp as gp
import revrand.legacygp.kernels as kern
from revrand import regression as reg
# from revrand import glm as reg
# from revrand.likelihoods import Gaussian
from revrand.basis_functions import LinearBasis, RandomRBF_ARD, RandomRBF
from yavanna.unsupervised.transforms import whiten, whiten_apply, standardise,\
    standardise_apply
from sklearn import linear_model, svm
from scipy.misc import imsave
import simplekml


def main():

    logging.basicConfig(level=logging.INFO)
    # log = logging.getLogger(__name__)

    # Settings
    truncate = 15
    label = 'Na_ppm_i_1'
    nfeatures = 400
    makekml = True

    y = np.load("y.npy")
    X = np.load("X.npy")
    Xs = np.load("Xstar.npy")
    label_names = np.load("label_names.npy")
    lons = np.load("lons.npy")
    lats = np.load("lats.npy")
    label_coords = np.load("label_coords.npy")
    x_bands = np.load("x_bands.npy")
    cube = np.load("cube.npy")

    labinds = {v: k for k, v in enumerate(label_names)}
    bandinds = {v: k for k, v in enumerate(x_bands)}

    # import IPython; IPython.embed(); exit()

    # Remove Lat-Lons and other layers from data
    bandmask = np.ones(X.shape[1], dtype=bool)
    bandmask[1] = False  # Lat/lon
    bandmask[2] = False  # Lat/lon
    bandmask[6] = False  # A lot of NaNs
    bandmask[13] = False  # Super low res
    bandmask[33] = False  # Super outlier
    bandmask[43] = False  # Super outlier
    bandmask[44] = False  # Super outlier
    bandmask[45] = False  # Super outlier
    bandmask[46] = False  # Super outlier
    bandmask[47] = False  # Super outlier
    bandmask[48] = False  # Super outlier and noisy


    # NOTES on the x_bands
    # Layer 9 may be categorical
    # Layer 10 may be categorical
    # Layer 11 may be categorical
    # Layer 15 is binary
    # Layer 18 may be categorical
    # Layer 42 may be categorical
    # Layer 51 may be categorical

    X = X[:, bandmask]
    Xs = Xs[:, bandmask]
    d = X.shape[1]

    # Remove NaNs in training data
    nanmask = np.sum(np.isnan(X), axis=1) >= 1
    Xf = X[~nanmask]
    yf = y[~nanmask]
    y_demo = yf[:, labinds[label]]
    # y_demo = Xf[:, bandinds[b'PM_Aster_ferrousIron_band_1']]
    # y_demo = Xf[:, bandinds[b'LOC_longs_band_1']]
    y_demo -= y_demo.mean()
    label_coordsf = label_coords[~nanmask]
    N = Xf.shape[0]

    # Remove NaNs in testing data
    nanmask_s = np.sum(np.isnan(Xs), axis=1) >= 1
    Xsf = Xs[~nanmask_s]
    Ns = Xsf.shape[0]
    Ns_nan = Xs.shape[0]

    # Whiten
    # Xs_w = Xsf
    # X_w = Xf

    # Xs_w, U, l, Xs_w_mean = whiten(Xsf, reducedims=truncate)
    # X_w = whiten_apply(Xf, U, l, Xs_w_mean)

    Xs_scale, Xmean, Xstd = standardise(Xsf)
    Xs_w, U, l, Xs_w_mean = whiten(Xs_scale, reducedims=truncate)
    X_w = whiten_apply(standardise_apply(Xf, Xmean, Xstd), U, l, Xs_w_mean)

    # Xs_w, Xmean, Xstd = standardise(Xsf)
    # X_w = standardise_apply(Xf, Xmean, Xstd)

    # X_scale, Xmean, Xstd = standardise(Xf)
    # X_w = X_scale
    # Xs_w = standardise_apply(Xsf, Xmean, Xstd)

    # X_scale, Xmean, Xstd = standardise(Xf)
    # X_w, U, l, X_w_mean = whiten(X_scale, reducedims=truncate)
    # Xs_w = whiten_apply(standardise_apply(Xsf, Xmean, Xstd), U, l, X_w_mean)

    D = Xs_w.shape[1]

    # Train
    lenscale = 1
    # lenlower = 1e-1
    # lenupper = 1e2

    # def kdef(h, k):
    #     return (h(1e-5, 1e2, 0.5)
    #             * k(kern.gaussian, [h(lenlower, lenupper, lenscale) for _ in
    #                                 range(D)])
    #             + k(kern.lognoise, h(-4, 1, -3)))

    # hyper_params = gp.learn(X_w, y_demo, kdef, verbose=True,
    #                         ftol=1e-15, maxiter=1000)

    # regressor = gp.condition(X_w, y_demo, kdef, hyper_params)

    # like = Gaussian()
    # lparams = [1.]
    # basis = RandomRBF_ARD(Xdim=X_w.shape[1], nbases=nfeatures)
    # hypers = np.ones(X_w.shape[1])
    # params = reg.learn(X_w, y_demo, like, lparams, basis,
    #                    hypers, verbose=True)
    # Ey, Vf, _, _ = reg.predict_meanvar(Xs_w, like, basis, *params)
    # Vy_g = Vf + params[2][0]
    # Sy_g = np.sqrt(Vy_g)
    # import IPython; IPython.embed()

    # basis = LinearBasis(onescol=True)
    # hypers = []
    # basis = LinearBasis(onescol=True) + RandomRBF(Xdim=D, nbases=nfeatures)
    # hypers = [lenscale]
    basis = RandomRBF_ARD(Xdim=D, nbases=nfeatures)
    hypers = np.ones(D) * lenscale
    # basis = RandomRBF(Xdim=D, nbases=nfeatures)
    # hypers = [lenscale]
    params = reg.learn(X_w, y_demo, basis, hypers, verbose=True, ftol=1e-6,
                       var=1)

    # lm = linear_model.BayesianRidge()
    # lm.fit(X_w, y_demo)
    # Ey = lm.predict(Xs_w)
    # svr = svm.SVR()
    # svr.fit(X_w, y_demo)
    # Ey = svr.predict(Xs_w)

    # Plotting
    # pl.bar(range(len(l)), l)

    # Prediction
    Ey = np.zeros(Ns)
    Vf = np.zeros(Ns)
    Vy = np.zeros(Ns)
    for inds in np.array_split(range(Ns), 10000):
        # query = gp.query(regressor, Xs_w[inds])
        # Ey[inds] = gp.mean(query)
        # Vy[inds] = gp.variance(query, noise=True)

        Ey[inds], Vf[inds], Vy[inds] = reg.predict(Xs_w[inds], basis, *params)

    # Sy = np.sqrt(Vy)

    Xs_w_nan = np.zeros((Ns_nan, D)) * np.NaN
    Xs_w_nan[~nanmask_s, :] = Xs_w
    px_lons = (lons[-1] - lons[0]) / float(lons.shape[0])
    px_lats = (lats[-1] - lats[0]) / float(lats.shape[0])
    extent = (lons[0] - .5 * px_lons, lons[-1] + .5 * px_lons,
              lats[0] - .5 * px_lats, lats[-1] + .5 * px_lats)

    # pl.figure()
    eigs = slice(0, 3)
    Xs_w_first3 = Xs_w[:, eigs] - Xs_w[:, eigs].min(axis=0)
    Xs_w_first3 /= (Xs_w[:, eigs].max(axis=0) - Xs_w[:, eigs].min(axis=0))
    Xs_w_first3_nan = np.zeros((Ns_nan, 4))
    Xs_w_first3_nan[~nanmask_s, 0:3] = Xs_w_first3
    Xs_w_first3_nan[~nanmask_s, 3] = 1.
    imshape = (lons.shape[0], lats.shape[0], -1)
    Im = np.flipud(np.transpose(np.reshape(Xs_w_first3_nan, imshape),
                                axes=[1, 0, 2]))
    if makekml:
        imsave('PCAcomponents.png', Im)

    pl.imshow(Im, extent=extent, interpolation='none')

    pl.figure()
    Ey_nan = np.zeros(Ns_nan) * np.NaN
    Ey_nan[~nanmask_s] = Ey
    EyIm = np.reshape(Ey_nan, imshape[0:2])
    # import IPython; IPython.embed()
    pl.imshow(EyIm.T, origin='lower', extent=extent, interpolation='none')
    pl.scatter(label_coordsf[:, 0], label_coordsf[:, 1],
               c=y_demo)
    pl.show()

    # Make KML files
    if makekml:
        kml = simplekml.Kml()
        ground = kml.newgroundoverlay(name='PCAcomponents')
        ground.icon.href = 'files/PCAcomponents.png'
        ground.latlonbox.north = lats[0]
        ground.latlonbox.south = lats[-1]
        ground.latlonbox.east = lons[-1]
        ground.latlonbox.west = lons[0]
        kml.addfile("PCAcomponents.png")
        kml.savekmz("PCAcomponents.kmz", format=False)

if __name__ == "__main__":
    main()
