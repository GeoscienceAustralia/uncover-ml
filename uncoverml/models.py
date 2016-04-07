""" Model Spec Objects and ML algorithm serialisation. """

import numpy as np

from revrand import regression
from revrand import glm
from revrand.basis_functions import LinearBasis, RandomRBF, RandomRBF_ARD
from revrand.likelihoods import Gaussian, Bernoulli, Poisson

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class LinearReg:

    def __init__(self, var=1., regulariser=1., diagcov=False, tol=1e-6,
                 maxit=500, verbose=True):

        self.params = {'basis': None,
                       'bparams': [],
                       'var': var,
                       'regulariser': regulariser,
                       'diagcov': diagcov,
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


class GaussianProcess(LinearReg):

    def __init__(self, nbases=500, lenscale=1., ard=False, *args, **kwargs):

        super(GaussianProcess, self).__init__(*args, **kwargs)

        self.nbases = nbases
        self.lenscale = lenscale
        self.ard = ard

    def _make_basis(self, X):

        d = X.shape[1]

        if self.ard:
            self.params['basis'] = RandomRBF_ARD(nbases=self.nbases, Xdim=d)
            self.params['bparams'] = [self.lenscale * np.ones(d)]
        else:
            self.params['basis'] = RandomRBF(nbases=self.nbases, Xdim=d)
            self.params['bparams'] = [self.lenscale]


class GenLinMod(GaussianProcess):

    def __init__(self, likelihood="Gaussian", lparams=[1.], postcomp=10,
                 use_sgd=True, batchsize=100, *args, **kwargs):

        super(GenLinMod, self).__init__(*args, **kwargs)

        # Extra params
        self.params['likelihood'] = lhoodmaps[likelihood]()
        self.params['lparams'] = lparams
        self.params['postcomp'] = postcomp
        self.params['batchsize'] = batchsize
        self.params['use_sgd'] = use_sgd

        # translate the parameters
        del self.params['diagcov']
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

        Ey, Vy, Ey_min, Ey_max = glm.predict_meanvar(X, *args)

        if uncertainty and (interval is not None):
            l, u = glm.predict_interval(interval, X, *args)

        return (Ey, Vy if interval is None else l, u) if uncertainty else Ey


lhoodmaps = {'Gaussian': Gaussian,
             'Bernoulli': Bernoulli,
             'Poisson': Poisson
             }

modelmaps = {'randomforest': RandomForestRegressor,
             'bayesreg': LinearReg,
             'gp': GaussianProcess,
             'svr': SVR,
             'glm': GenLinMod
             }
