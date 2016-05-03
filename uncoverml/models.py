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

        Phi = self.base(X, *self.bparams)
        pCp = [p.dot(self.C).dot(p.T) for p in Phi]
        return 0.5 * (np.log(self.optvar + np.array(pCp))
                      + np.log(self.optvar))

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
