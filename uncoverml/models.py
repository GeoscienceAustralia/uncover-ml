""" Model Spec Objects and ML algorithm serialisation. """

import numpy as np

from revrand import regression
from revrand import glm
from revrand.basis_functions import LinearBasis, RandomRBF, RandomRBF_ARD
from revrand.likelihoods import Gaussian, Bernoulli, Poisson
from revrand.btypes import Parameter, Positive

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class LinearReg(object):

    def __init__(self, var=Parameter(1., Positive()),
                 regulariser=Parameter(1., Positive()), tol=1e-6, maxit=500,
                 verbose=True):

        self.params = {'basis': None,
                       'var': var,
                       'regulariser': regulariser,
                       'tol': tol,
                       'maxit': maxit,
                       'verbose': verbose,
                       }

    def fit(self, X, y):

        self._make_basis(X)
        m, C, bparams, var = regression.learn(X, y, **self.params)
        self.params['m'] = m
        self.params['C'] = C
        self.params['bparams'] = bparams
        self.params['var'] = var

        return self

    def predict(self, X, uncertainty=False):

        Ey, Vf, Vy = regression.predict(X,
                                        self.params['basis'],
                                        self.params['m'],
                                        self.params['C'],
                                        self.params['bparams'],
                                        self.params['var']
                                        )

        return (Ey, Vf, Vy) if uncertainty else Ey

    def get_params(self):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self

    def _make_basis(self, X):

        self.params['basis'] = LinearBasis(onescol=True)


class ApproxGP(LinearReg):

    def __init__(self, nbases=200, lenscale=1., ard=False, *args, **kwargs):

        super(ApproxGP, self).__init__(*args, **kwargs)

        self.nbases = nbases
        self.lenscale = lenscale
        self.ard = ard

    def _make_basis(self, X):

        d = X.shape[1]

        if self.ard:
            lenard = self.lenscale * np.ones(d)
            self.params['basis'] = \
                RandomRBF_ARD(nbases=self.nbases, Xdim=d,
                              lenscale_init=Parameter(lenard, Positive()))
        else:
            self.params['basis'] = \
                RandomRBF(nbases=self.nbases, Xdim=d,
                          lenscale_init=Parameter(self.lenscale, Positive()))


class GenLinMod(ApproxGP):

    def __init__(self, likelihood="Gaussian", lparams=Parameter(1, Positive()),
                 postcomp=10, use_sgd=True, batchsize=100, maxit=100, *args,
                 **kwargs):

        super(GenLinMod, self).__init__(*args, **kwargs)

        # Extra params
        self.params['likelihood'] = lhoodmaps[likelihood](lparams)
        self.params['postcomp'] = postcomp
        self.params['batchsize'] = batchsize
        self.params['maxit'] = maxit
        self.params['use_sgd'] = use_sgd

        # translate the parameters
        del self.params['var']

    def _make_basis(self, X):

        super(GenLinMod, self)._make_basis(X)
        self.params['basis'] += LinearBasis(onescol=True)

    def fit(self, X, y):

        self._make_basis(X)
        m, C, lparams, bparams = glm.learn(X, y, **self.params)
        self.params['m'] = m
        self.params['C'] = C
        self.params['bparams'] = bparams
        self.params['lparams'] = lparams

        return self

    def predict(self, X, uncertainty=False, interval=None):

        args = [self.params['likelihood'],
                self.params['basis'],
                self.params['m'],
                self.params['C'],
                self.params['lparams'],
                self.params['bparams']
                ]

        Ey, Vy, Ey_min, Ey_max = glm.predict_moments(X, *args)

        if uncertainty and (interval is not None):
            l, u = glm.predict_interval(interval, X, *args)

        return (Ey, Vy if interval is None else l, u) if uncertainty else Ey


modelmaps = {'randomforest': RandomForestRegressor,
             'bayesreg': LinearReg,
             'approxgp': ApproxGP,
             'svr': SVR,
             'glm': GenLinMod
             }


lhoodmaps = {'Gaussian': Gaussian,
             'Bernoulli': Bernoulli,
             'Poisson': Poisson
             }
