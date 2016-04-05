""" Model Spec Objects and ML algorithm serialisation. """

# import importlib

from revrand import regression
from revrand.basis_functions import LinearBasis, RandomRBF

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class LinearReg:

    def __init__(self, var=1., regulariser=1., diagcov=False, ftol=1e-6,
                 maxit=1000, verbose=True):

        self.params = {'basis': None,
                       'bparams': [],
                       'var': var,
                       'regulariser': regulariser,
                       'diagcov': diagcov,
                       'ftol': ftol,
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

    def _make_basis(self, X):

        self.params['basis'] = LinearBasis(onescol=True)


class GaussianProcess(LinearReg):

    def __init__(self, nbases=500, lenscale=1., *args, **kwargs):

        super(GaussianProcess, self).__init__(*args, **kwargs)

        self.nbases = nbases
        self.params['bparams'] = [lenscale]

    def _make_basis(self, X):

        self.params['basis'] = RandomRBF(nbases=self.nbases, Xdim=X.shape[1])


modelmaps = {'randomforest': RandomForestRegressor,
             'bayesreg': LinearReg,
             'gp': GaussianProcess,
             'svr': SVR
             }
