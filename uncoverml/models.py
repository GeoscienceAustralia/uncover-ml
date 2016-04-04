""" Model Spec Objects and ML algorithm serialisation. """

# import importlib

from revrand import regression
from revrand.basis_functions import LinearBasis

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class RevrandReg:

    def __init__(self, basis=LinearBasis(onescol=False), bparams=[], var=1.,
                 regulariser=1., diagcov=False, ftol=1e-6, maxit=1000,
                 verbose=True):

        self.params = {'basis': basis,
                       'bparams': bparams,
                       'var': var,
                       'regulariser': regulariser,
                       'diagcov': diagcov,
                       'ftol': ftol,
                       'maxit': maxit,
                       'verbose': verbose,
                       }

    def fit(self, X, y):

        m, C, bparams, var = regression.learn(X, y, **self.params)
        self.params['m'] = m
        self.params['C'] = C
        self.params['bparams'] = bparams
        self.params['var'] = var

        return self

    def predict(self, X):

        return regression.predict(X,
                                  self.params['basis'],
                                  self.params['m'],
                                  self.params['C'],
                                  self.params['bparams'],
                                  self.params['var']
                                  )

    def get_params(self):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self


modelmaps = {'randomforest': RandomForestRegressor,
             'bayesreg': RevrandReg,
             'svr': SVR
             }
