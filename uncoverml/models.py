"""Model Spec Objects and ML algorithm serialisation."""

import numpy as np

from revrand import StandardLinearModel
from revrand.basis_functions import LinearBasis, BiasBasis, RandomRBF, \
    RandomLaplace, RandomCauchy, RandomMatern32, RandomMatern52
from revrand.likelihoods import Gaussian, Bernoulli, Poisson
from revrand.btypes import Parameter, Positive
from revrand.utils import atleast_list

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

import uncoverml.transforms as transforms


class LinearModel(StandardLinearModel):

    def fit(self, X, y):

        self._make_basis(X)
        return super(LinearModel, self).fit(X, y)

    def predict_proba(self, X):

        Ey, _, Vy = super(LinearModel, self).predict_moments(X)
        return Ey, Vy

    def entropy_reduction(self, X):

        Phi = self.basis.transform(X, *atleast_list(self.hypers))
        pCp = [p.dot(self.covariance).dot(p.T) for p in Phi]
        MI = 0.5 * (np.log(self.var + np.array(pCp)) - np.log(self.var))
        return MI

    def _make_basis(self, X):

        pass


class LinearReg(LinearModel):

    def __init__(self, onescol=True, var=Parameter(1., Positive()),
                 regulariser=Parameter(1., Positive()), tol=1e-8,
                 maxiter=1000):

        basis = LinearBasis(onescol=onescol)
        super(LinearReg, self).__init__(basis=basis,
                                        var=var,
                                        regulariser=regulariser,
                                        tol=tol,
                                        maxiter=maxiter
                                        )


class ApproxGP(LinearModel):

    def __init__(self, kern='rbf', nbases=200, lenscale=.1,
                 var=Parameter(1., Positive()),
                 regulariser=Parameter(1., Positive()), tol=1e-8,
                 maxiter=1000):

        self.nbases = nbases
        self.lenscale = lenscale if np.isscalar(lenscale) \
            else np.asarray(lenscale)
        self.kern = kern

        super(ApproxGP, self).__init__(basis=None,
                                       var=var,
                                       regulariser=regulariser,
                                       tol=tol,
                                       maxiter=maxiter
                                       )

    def _make_basis(self, X):

        self.basis = basismap[self.kern](Xdim=X.shape[1],
                                         nbases=self.nbases,
                                         lenscale_init=Parameter(self.lenscale,
                                                                 Positive())
                                         ) + BiasBasis()


class RandomForestRegressor(RFR):
    """
    Implements a "probabilistic" output by looking at the variance of the
    decision tree estimator ouputs.
    """

    def predict_proba(self, X, *args):
        Ey = self.predict(X)

        Vy = np.zeros_like(Ey)
        for dt in self.estimators_:
            Vy += (dt.predict(X, *args) - Ey)**2

        Vy /= len(self.estimators_)

        return Ey, Vy


#
# Transformer factory
#

def transform_targets(Learner):

    class TransformedLearner(Learner):

        def __init__(self, transform='indentity', *args, **kwargs):

            super(TransformedLearner, self).__init__(*args, **kwargs)
            self.ytform = transforms.transforms[transform]()

        def fit(self, X, y, *args, **kwargs):

            self.ytform.fit(y)
            y_t = self.ytform.transform(y)

            return super(TransformedLearner, self).fit(X, y_t, *args, **kwargs)

        def predict(self, X, *args):

            Ey_t = super(TransformedLearner, self).predict(X, *args)
            Ey = self.ytform.itransform(Ey_t)

            return Ey

        if hasattr(Learner, 'predict_proba'):
            def predict_proba(self, X, *args):

                Ey_t, Vy_t = super(TransformedLearner, self).predict(X, *args)

                nsamples = 100
                Ey = self.ytform.itransform(Ey_t)
                Sy_t = np.sqrt(Vy_t)
                Vy = np.var([Ey_t + np.random.randn(Ey.shape) * Sy_t
                             for _ in range(nsamples)], axis=1)

                return Ey, Vy

    return TransformedLearner


# These are purely so we can pickle

class SVR_Transformed(transform_targets(SVR)):
    pass


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
    unstackfunc = lambda catdata, *nargs: \
        func(*(unstack(catdata) + list(nargs)))

    return apply_masked(unstackfunc, np.ma.hstack(datastack), args)


#
# Static module properties
#

modelmaps = {'randomforest': transform_targets(RandomForestRegressor),
             'bayesreg': transform_targets(LinearReg),
             'approxgp': transform_targets(ApproxGP),
             'svr': SVR_Transformed,
             'kernelridge': transform_targets(KernelRidge),
             'ardregression': transform_targets(ARDRegression),
             'decisiontree': transform_targets(DecisionTreeRegressor),
             'extratree': transform_targets(ExtraTreeRegressor)
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
