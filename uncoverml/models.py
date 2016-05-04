""" Model Spec Objects and ML algorithm serialisation. """

import numpy as np

from revrand import regression
from revrand.basis_functions import LinearBasis, RandomRBF, RandomLaplace, \
    RandomCauchy, RandomMatern32, RandomMatern52
from revrand.likelihoods import Gaussian, Bernoulli, Poisson
from revrand.btypes import Parameter, Positive

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class LinearModel(BaseEstimator):

    def __init__(self, basis, var=Parameter(1., Positive()),
                 regulariser=Parameter(1., Positive()), tol=1e-6, maxit=500,
                 verbose=True):

        self.basis = basis
        self.var = var
        self.regulariser = regulariser
        self.tol = tol
        self.maxit = maxit
        self.verbose = verbose

    def fit(self, X, y):

        self._make_basis(X)
        m, C, bparams, var = regression.learn(X, y,
                                              basis=self.basis,
                                              var=self.var,
                                              regulariser=self.regulariser,
                                              tol=self.tol,
                                              maxit=self.maxit,
                                              verbose=self.verbose
                                              )
        self.m = m
        self.C = C
        self.bparams = bparams
        self.optvar = var

        return self

    def predict(self, X, uncertainty=False):

        Ey, _, Vy = regression.predict(X,
                                       self.basis,
                                       self.m,
                                       self.C,
                                       self.bparams,
                                       self.optvar
                                       )

        return (Ey, Vy) if uncertainty else Ey

    def entropy_reduction(self, X):

        Phi = self.basis(X, *self.bparams)
        pCp = [p.dot(self.C).dot(p.T) for p in Phi]
        return 0.5 * (np.log(self.optvar + np.array(pCp))
                      - np.log(self.optvar))

    def _make_basis(self, X):

        pass


class LinearReg(LinearModel):

    def __init__(self, onescol=True, var=Parameter(1., Positive()),
                 regulariser=Parameter(1., Positive()), tol=1e-6, maxit=500,
                 verbose=True):

        basis = LinearBasis(onescol=onescol)
        super(LinearReg, self).__init__(basis, var, regulariser, tol, maxit,
                                        verbose)


class ApproxGP(LinearModel):

    def __init__(self, kern='rbf', nbases=200, lenscale=1.,
                 var=Parameter(1., Positive()),
                 regulariser=Parameter(1., Positive()), tol=1e-6, maxit=500,
                 verbose=True):

        super(ApproxGP, self).__init__(None, var, regulariser, tol,
                                       maxit, verbose)

        self.nbases = nbases
        self.lenscale = lenscale
        self.kern = kern

    def _make_basis(self, X):

        self.basis = basismap[self.kern](Xdim=X.shape[1], nbases=self.nbases,
                                         lenscale_init=Parameter(self.lenscale,
                                                                 Positive()))


#
# Helper functions
#

def apply_masked(func, data, args=()):

    # No masked data
    if np.ma.count_masked(data) == 0:
        return func(data.data, *args)

    # Prediction with missing inputs
    okdata = (data.mask.sum(axis=1)) == 0 if data.ndim == 2 else ~data.mask
    res = func(data.data[okdata], *args)

    # For training/fitting that returns nothing
    if not isinstance(res, np.ndarray):
        return res

    # Fill in a padded array the size of the original
    mres = np.empty(len(data)) if res.ndim == 1 \
        else np.empty((len(data), res.shape[1]))
    mres[okdata] = res

    # Make sure the mask is consistent with the original array
    mask = ~okdata
    if mres.ndim > 1:
        mask = np.tile(mask, (mres.shape[1], 1)).T

    return np.ma.array(mres, mask=mask)


def apply_multiple_masked(func, data, args=()):

    datastack = []
    dims = []
    flat = []
    for d in data:
        if d.ndim == 2:
            datastack.append(d)
            dims.append(d.shape[1])
            flat.append(False)
        elif d.ndim == 1:
            datastack.append(d[:, np.newaxis])
            dims.append(1)
            flat.append(True)
        else:
            raise RuntimeError("data arrays have to be 1 or 2D arrays")

    # Decorate functions to work on stacked data
    dims = np.cumsum(dims[:-1])  # dont split by last dim
    unstack = lambda catdata: [d.flatten() if f else d for d, f
                               in zip(np.hsplit(catdata, dims), flat)]
    unstackfunc = lambda catdata, *nargs: func(*unstack(catdata), *nargs)

    return apply_masked(unstackfunc, np.ma.hstack(datastack), args)


#
# Static module properties
#

modelmaps = {'randomforest': RandomForestRegressor,
             'bayesreg': LinearReg,
             'approxgp': ApproxGP,
             'svr': SVR,
             }


lhoodmaps = {'Gaussian': Gaussian,
             'Bernoulli': Bernoulli,
             'Poisson': Poisson
             }

basismap = {'rbf': RandomRBF,
            'laplace': RandomLaplace,
            'cauchy': RandomCauchy,
            'matern32': RandomMatern32,
            'matern52': RandomMatern52
            }

probmodels = (LinearReg, ApproxGP, LinearModel)
probmodels_str = ['approxgp', 'bayesreg']
