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
from sklearn import linear_model


def main():

    logging.basicConfig(level=logging.INFO)
    # log = logging.getLogger(__name__)


    # Settings
    truncate = 50
    label = 'Al_ppm_imp'
    nfeatures = 200

    y = np.load("y.npy")
    X = np.load("X.npy")
    Xs = np.load("Xstar.npy")
    label_names = np.load("label_names.npy")
    lons = np.load("lons.npy")
    lats = np.load("lats.npy")
    label_coords = np.load("label_coords.npy")
    # cube = np.load("cube.npy")

    labinds = {v: k for k, v in enumerate(label_names)}

    # Remove NaNs in training data
    nanmask = np.sum(np.isnan(X), axis=1) >= 1
    Xf = X[~nanmask]
    yf = y[~nanmask]
    label_coordsf = label_coords[~nanmask]

    # Remove NaNs in testing data
    nanmask_s = np.sum(np.isnan(Xs), axis=1) >= 1
    Xsf = Xs[~nanmask_s]

    # Whiten
    # Xs_w = Xsf
    # X_w = Xf

    # Xs_scale, Xmean, Xstd = standardise(Xsf)
    # Xs_w, U, l, Xs_w_mean = whiten(Xs_scale, reducedims=truncate)
    # X_w = whiten_apply(standardise_apply(Xf, Xmean, Xstd), U, l, Xs_w_mean)

    # X_scale, Xmean, Xstd = standardise(Xf)
    # X_w = X_scale
    # Xs_w = standardise_apply(Xsf, Xmean, Xstd)

    X_scale, Xmean, Xstd = standardise(Xf)
    X_w, U, l, X_w_mean = whiten(X_scale, reducedims=truncate)
    Xs_w = whiten_apply(standardise_apply(Xsf, Xmean, Xstd), U, l, X_w_mean)

    # Train
    # lenscale = 1.

    # def kdef(h, k):
    #     return (h(1e-5, 1e2, 0.5) * k(kern.gaussian, h(1e-1, 1e2, lenscale)) +
    #             k(kern.lognoise, h(-4, 1, -3)))

    # hyper_params = gp.learn(X_w, yf[:, labinds[label]], kdef, verbose=True,
    #                         ftol=1e-15, maxiter=1000)

    # regressor = gp.condition(X_w, yf[:, labinds[label]], kdef, hyper_params)
    # query = gp.query(regressor, Xs_w)
    # Ey = gp.mean(query)
    # Vf = gp.variance(query)
    # Vy = gp.variance(query, noise=True)
    # Sy = np.sqrt(Vy)

    # like = Gaussian()
    # lparams = [1.]
    # basis = RandomRBF_ARD(Xdim=X_w.shape[1], nbases=nfeatures)
    # hypers = np.ones(X_w.shape[1])
    # params = reg.learn(X_w, yf[:, labinds[label]], like, lparams, basis,
    #                    hypers, verbose=True)
    # Ey, Vf, _, _ = reg.predict_meanvar(Xs_w, like, basis, *params)
    # Vy_g = Vf + params[2][0]
    # Sy_g = np.sqrt(Vy_g)
    # import IPython; IPython.embed()

    # basis = LinearBasis(onescol=False)
    # hypers = []
    # basis = RandomRBF_ARD(Xdim=X_w.shape[1], nbases=nfeatures)
    # hypers = np.ones(X_w.shape[1])
    # basis = RandomRBF(Xdim=X_w.shape[1], nbases=nfeatures)
    # hypers = [1.]
    # params = reg.learn(X_w, yf[:, labinds[label]], basis, hypers, verbose=True)
    # Ey, Vf, Vy = reg.predict(Xs_w, basis, *params)

    lm = linear_model.BayesianRidge()
    lm.fit(X_w, yf[:, labinds[label]])
    Ey = lm.predict(Xs_w)


    # Plotting
    # pl.bar(range(len(l)), l)

    Xs_w_nan = np.zeros((Xs.shape[0], Xs_w.shape[1])) * np.NaN
    Xs_w_nan[~nanmask_s, :] = Xs_w
    px_lons = (lons[-1] - lons[0]) / float(lons.shape[0])
    px_lats = (lats[-1] - lats[0]) / float(lats.shape[0])
    extent = (lons[0] - .5 * px_lons, lons[-1] + .5 * px_lons,
              lats[0] - .5 * px_lats, lats[-1] + .5 * px_lats)

    # pl.figure()
    eigs = slice(1, 4)
    Xs_w_first3 = Xs_w[:, eigs] - Xs_w[:, eigs].min(axis=0)
    Xs_w_first3 /= (Xs_w[:, eigs].max(axis=0) - Xs_w[:, eigs].min(axis=0))
    Xs_w_first3_nan = np.zeros((Xs.shape[0], 4))
    Xs_w_first3_nan[~nanmask_s, 0:3] = Xs_w_first3
    Xs_w_first3_nan[~nanmask_s, 3] = 1.
    imshape = (lons.shape[0], lats.shape[0], -1)
    Im = np.reshape(Xs_w_first3_nan, imshape)

    pl.imshow(np.transpose(Im, axes=[1, 0, 2]), origin='lower', extent=extent,
              interpolation='none')

    # pl.imshow(Im[:, :, 0].T, origin='lower', extent=extent,
    #           interpolation='none')
    # pl.imshow(Im[:, :, 8].T, origin='upper', extent=extent,
    #           interpolation='none')
    # pl.scatter(lons[0] * np.ones_like(lats), lats)

    pl.figure()

    Ey_nan = np.zeros(Xs.shape[0]) * np.NaN
    Ey_nan[~nanmask_s] = Ey
    EyIm = np.reshape(Ey_nan, imshape[0:2])
    # import IPython; IPython.embed()
    pl.imshow(EyIm.T, origin='lower', extent=extent, interpolation='none')
    pl.scatter(label_coordsf[:, 0], label_coordsf[:, 1],
               c=yf[:, labinds[label]])
    pl.show()


if __name__ == "__main__":
    main()
