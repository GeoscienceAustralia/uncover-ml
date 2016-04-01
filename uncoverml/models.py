""" Model Spec Objects and ML algorithm serialisation. """

import importlib

from revrand import regression
from revrand.basis_functions import LinearBasis


modelmaps = {'randomforest': 'sklearn.ensemble.RandomForestRegressor',
             'bayesreg': 'uncoverml.models.RevReg',
             'svr': 'sklearn.svm.SVR'
             }


class RevReg:

    def __init__(self, basis=LinearBasis(onescol=True), bparams=[], var=1.,
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
                       'm': None,
                       'C': None
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


class ModelSpec:

    def __init__(self, importpath, modelname, **params):

        self.importpath = importpath
        self.modelname = modelname
        self.params = params

    def to_dict(self):

        return {'importpath': self.importpath,
                'modelname': self.modelname,
                'parameters': self.params
                }

    @classmethod
    def from_dict(cls, mod_dict):

        return cls(mod_dict['importpath'], mod_dict['modelname'],
                   **mod_dict['params'])


def learn_model(X, y, modelspec, *args, **kwargs):

    mod = importlib.import_module(modelspec.importpath)
    mlobj = getattr(mod, modelspec.modelname)(*args, **kwargs)
    mlobj.fit(X, y)
    modelspec.params = mlobj.get_params()
    return modelspec


def predict_model(X, modelspec):

    mod = importlib.import_module(modelspec.importpath)
    mlobj = getattr(mod, modelspec.modelname)()
    mlobj = mlobj.set_params(**modelspec.params)
    return mlobj.predict(X)
