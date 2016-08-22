"""Model Objects and ML algorithm serialisation."""

from itertools import chain

import numpy as np
from scipy.stats import norm

from revrand import StandardLinearModel, GeneralisedLinearModel
from revrand.basis_functions import LinearBasis, BiasBasis, RandomRBF, \
    RandomLaplace, RandomCauchy, RandomMatern32, RandomMatern52
from revrand.likelihoods import Gaussian
from revrand.btypes import Parameter, Positive
from revrand.utils import atleast_list
from revrand.optimize import Adam

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

import uncoverml.transforms as transforms
from uncoverml.likelihoods import Switching
from uncoverml.cubist import Cubist


#
# Mixin classes for providing pipeline compatibility to revrand
#

class BasisMakerMixin():

    def fit(self, X, y, *args, **kwargs):

        self._make_basis(X)
        return super().fit(X, y, *args, **kwargs)  # args for GLMs

    def _store_params(self, kern, nbases, lenscale, ard):

        self.kern = kern
        self.nbases = nbases
        self.ard = ard
        self.lenscale = lenscale if np.isscalar(lenscale) \
            else np.asarray(lenscale)

    def _make_basis(self, X):

        D = X.shape[1]
        lenscale = self.lenscale
        if self.ard and D > 1:
            lenscale = np.ones(D) * lenscale

        lenscale_init = Parameter(lenscale, Positive())
        gpbasis = basismap[self.kern](Xdim=X.shape[1], nbases=self.nbases,
                                      lenscale_init=lenscale_init)

        self.basis = gpbasis + BiasBasis()


class PredictProbaMixin():
    """
    Mixin class for providing a ``predict_proba`` method to the
    StandardLinearModel class in revrand.
    """

    def predict_proba(self, X, interval=0.95, *args, **kwargs):
        """
        Predictive mean and variance for a probabilistic regressor.

        Parameters
        ----------
        X: ndarray
            (Ns, d) array query dataset (Ns samples, d dimensions).
        interval: float, optional
            The percentile confidence interval (e.g. 95%) to return.
        fields: dict, optional
            dictionary of fields parsed from the shape file.
            ``indicator_field`` should be a key in this dictionary. If this is
            not present, then a Gaussian likelihood will be used for all
            predictions. The only time this may be input if for cross
            validation.

        Returns
        -------
        Ey: ndarray
            The expected value of ys for the query inputs, X of shape (Ns,).
        Vy: ndarray
            The expected variance of ys (excluding likelihood noise terms) for
            the query inputs, X of shape (Ns,).
        ql: ndarray
            The lower end point of the interval with shape (Ns,)
        qu: ndarray
            The upper end point of the interval with shape (Ns,)
        """

        Ey, Vy = self.predict_moments(X, *args, **kwargs)
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


class GLMPredictProbaMixin():
    """
    Mixin class for providing a ``predict_proba`` method to the
    GeneralisedLinearModel class in revrand.

    This is especially for use with Gaussian likelihood models.
    """

    def predict_proba(self, X, interval=0.95, *args, **kwargs):
        """
        Predictive mean and variance for a probabilistic regressor.

        Parameters
        ----------
        X: ndarray
            (Ns, d) array query dataset (Ns samples, d dimensions).
        interval: float, optional
            The percentile confidence interval (e.g. 95%) to return.
        fields: dict, optional
            dictionary of fields parsed from the shape file.
            ``indicator_field`` should be a key in this dictionary. If this is
            not present, then a Gaussian likelihood will be used for all
            predictions. The only time this may be input if for cross
            validation.

        Returns
        -------
        Ey: ndarray
            The expected value of ys for the query inputs, X of shape (Ns,).
        Vy: ndarray
            The expected variance of ys (excluding likelihood noise terms) for
            the query inputs, X of shape (Ns,).
        ql: ndarray
            The lower end point of the interval with shape (Ns,)
        qu: ndarray
            The upper end point of the interval with shape (Ns,)
        """

        Ey, Vy = self.predict_moments(X, *args, **kwargs)
        Vy += self.like_hypers
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


class MutualInfoMixin():

    def entropy_reduction(self, X):

        Phi = self.basis.transform(X, *atleast_list(self.hypers))
        pCp = [p.dot(self.covariance).dot(p.T) for p in Phi]
        MI = 0.5 * (np.log(self.var + np.array(pCp)) - np.log(self.var))
        return MI


class TagsMixin():
    """
    Mixin class to aid a pipeline in establishing the types of predictive
    outputs to be expected from the ML algorithms in this module.
    """

    def get_predict_tags(self):
        """
        Get the types of prediction outputs from this algorithm.

        Returns
        -------
        list:
            of strings with the types of outputs that can be returned by this
            algorithm. This depends on the prediction methods implemented (e.g.
            ``predict``, `predict_proba``, ``entropy_reduction``).
        """

        tags = ['Prediction']
        if hasattr(self, 'predict_proba'):
            tags.extend(['Variance', 'Lower quantile', 'Upper quantile'])

        if hasattr(self, 'entropy_reduction'):
            tags.append('Expected reduction in entropy')

        return tags


#
# Specialisation of revrand's interface to work from the command line with a
# few curated algorithms
#

class LinearReg(StandardLinearModel, PredictProbaMixin, MutualInfoMixin):
    """
    Bayesian standard linear model.

    Parameters
    ----------
    onescol: bool, optional
        If true, prepend a column of ones onto X (i.e. a bias term)
    var: Parameter, optional
        observation variance initial value.
    regulariser: Parameter, optional
        weight regulariser (variance) initial value.
    tol: float, optional
        optimiser function tolerance convergence criterion.
    maxiter: int, optional
        maximum number of iterations for the optimiser.
    """

    def __init__(self, onescol=True, var=1., regulariser=1., tol=1e-8,
                 maxiter=1000):

        basis = LinearBasis(onescol=onescol)
        super().__init__(basis=basis,
                         var=Parameter(var, Positive()),
                         regulariser=Parameter(regulariser, Positive()),
                         tol=tol,
                         maxiter=maxiter
                         )


class ApproxGP(BasisMakerMixin, StandardLinearModel, PredictProbaMixin,
               MutualInfoMixin):
    """
    An approximate Gaussian process for medium scale data.

    Parameters
    ----------
    kern: str, optional
        the (approximate) kernel to use with this Gaussian process. Have a look
        at :code:`basismap` dictionary for appropriate kernel approximations.
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base).
        The higher this number, the more accurate the kernel approximation, but
        the longer the runtime of the algorithm. Usually if X is high
        dimensional, this will have to also be high dimensional.
    lenscale: float, optional
        the initial value for the kernel length scale to be learned.
    ard: bool, optional
        Whether to use a different length scale for each dimension of X or a
        single length scale. This will result in a longer run time, but
        potentially better results.
    var: Parameter, optional
        observation variance initial value.
    regulariser: Parameter, optional
        weight regulariser (variance) initial value.
    tol: float, optional
        optimiser function tolerance convergence criterion.
    maxiter: int, optional
        maximum number of iterations for the optimiser.
    """

    def __init__(self, kern='rbf', nbases=50, lenscale=1., var=1.,
                 regulariser=1., ard=True, tol=1e-8, maxiter=1000):

        super().__init__(basis=None,
                         var=Parameter(var, Positive()),
                         regulariser=Parameter(regulariser, Positive()),
                         tol=tol,
                         maxiter=maxiter
                         )

        self._store_params(kern, nbases, lenscale, ard)


class SGDLinearReg(GeneralisedLinearModel, GLMPredictProbaMixin):
    """
    Bayesian standard linear model, using stochastic gradients.

    This uses the Adam stochastic gradients algorithm;
    http://arxiv.org/pdf/1412.6980

    Parameters
    ----------
    onescol: bool, optional
        If true, prepend a column of ones onto X (i.e. a bias term)
    var: Parameter, optional
        observation variance initial value.
    regulariser: Parameter, optional
        weight regulariser (variance) initial value.
    maxiter: int, optional
        Number of iterations to run for the stochastic gradients algorithm.
    batch_size: int, optional
        number of observations to use per SGD batch.
    alpha: float, optional
        stepsize to give the stochastic gradient optimisation update.
    beta1: float, optional
        smoothing/decay rate parameter for the stochastic gradient, must be
        [0, 1].
    beta2: float, optional
        smoothing/decay rate parameter for the squared stochastic gradient,
        must be [0, 1].
    epsilon: float, optional
        "jitter" term to ensure continued learning in stochastic gradients
        (should be small).
    random_state: int or RandomState, optional
        random seed

    Note
    ----
    Setting the ``random_state`` may be important for getting consistent
    looking predictions when many chunks/subchunks are used. This is because
    the predictive distribution is sampled for these algorithms!
    """

    def __init__(self, onescol=True, var=1., regulariser=1., maxiter=3000,
                 batch_size=10, alpha=0.01, beta1=0.9, beta2=0.99,
                 epsilon=1e-8, random_state=None):

        super().__init__(likelihood=Gaussian(Parameter(var, Positive())),
                         basis=LinearBasis(onescol),
                         regulariser=Parameter(regulariser, Positive()),
                         maxiter=maxiter,
                         batch_size=batch_size,
                         updater=Adam(alpha, beta1, beta2, epsilon),
                         random_state=random_state
                         )


class SGDApproxGP(BasisMakerMixin, GeneralisedLinearModel,
                  GLMPredictProbaMixin):
    """
    An approximate Gaussian process for large scale data using stochastic
    gradients.

    This uses the Adam stochastic gradients algorithm;
    http://arxiv.org/pdf/1412.6980

    Parameters
    ----------
    kern: str, optional
        the (approximate) kernel to use with this Gaussian process. Have a look
        at :code:`basismap` dictionary for appropriate kernel approximations.
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base).
        The higher this number, the more accurate the kernel approximation, but
        the longer the runtime of the algorithm. Usually if X is high
        dimensional, this will have to also be high dimensional.
    lenscale: float, optional
        the initial value for the kernel length scale to be learned.
    ard: bool, optional
        Whether to use a different length scale for each dimension of X or a
        single length scale. This will result in a longer run time, but
        potentially better results.
    var: Parameter, optional
        observation variance initial value.
    regulariser: Parameter, optional
        weight regulariser (variance) initial value.
    maxiter: int, optional
        Number of iterations to run for the stochastic gradients algorithm.
    batch_size: int, optional
        number of observations to use per SGD batch.
    alpha: float, optional
        stepsize to give the stochastic gradient optimisation update.
    beta1: float, optional
        smoothing/decay rate parameter for the stochastic gradient, must be
        [0, 1].
    beta2: float, optional
        smoothing/decay rate parameter for the squared stochastic gradient,
        must be [0, 1].
    epsilon: float, optional
        "jitter" term to ensure continued learning in stochastic gradients
        (should be small).
    random_state: int or RandomState, optional
        random seed

    Note
    ----
    Setting the ``random_state`` may be important for getting consistent
    looking predictions when many chunks/subchunks are used. This is because
    the predictive distribution is sampled for these algorithms!
    """

    def __init__(self, kern='rbf', nbases=50, lenscale=1., var=1.,
                 regulariser=1., ard=True, maxiter=3000, batch_size=10,
                 alpha=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8,
                 random_state=None):

        super().__init__(likelihood=Gaussian(Parameter(var, Positive())),
                         basis=None,
                         regulariser=Parameter(regulariser, Positive()),
                         maxiter=maxiter,
                         batch_size=batch_size,
                         updater=Adam(alpha, beta1, beta2, epsilon),
                         random_state=random_state
                         )

        self._store_params(kern, nbases, lenscale, ard)


# Bespoke regressor for basin-depth problems
class DepthRegressor(BasisMakerMixin, GeneralisedLinearModel, TagsMixin,
                     GLMPredictProbaMixin):
    """
    A specialised approximate Gaussian process for large scale data using
    stochastic gradients.

    This has been specialised for depth-measurments where some observations are
    merely constraints that the phenemena of interest is below the observed
    depth, and not measured directly. See the project report for details.

    This uses the Adam stochastic gradients algorithm;
    http://arxiv.org/pdf/1412.6980

    Parameters
    ----------
    kern: str, optional
        the (approximate) kernel to use with this Gaussian process. Have a look
        at :code:`basismap` dictionary for appropriate kernel approximations.
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base).
        The higher this number, the more accurate the kernel approximation, but
        the longer the runtime of the algorithm. Usually if X is high
        dimensional, this will have to also be high dimensional.
    lenscale: float, optional
        the initial value for the kernel length scale to be learned.
    ard: bool, optional
        Whether to use a different length scale for each dimension of X or a
        single length scale. This will result in a longer run time, but
        potentially better results.
    var: Parameter, optional
        observation variance initial value.
    regulariser: Parameter, optional
        weight regulariser (variance) initial value.
    indicator_field: str, optional
        The name of the field from the target shapefile (passed as a dict to
        fit and predict) that indicates whether an observation is censored or
        not.
    maxiter: int, optional
        Number of iterations to run for the stochastic gradients algorithm.
    batch_size: int, optional
        number of observations to use per SGD batch.
    alpha: float, optional
        stepsize to give the stochastic gradient optimisation update.
    beta1: float, optional
        smoothing/decay rate parameter for the stochastic gradient, must be
        [0, 1].
    beta2: float, optional
        smoothing/decay rate parameter for the squared stochastic gradient,
        must be [0, 1].
    epsilon: float, optional
        "jitter" term to ensure continued learning in stochastic gradients
        (should be small).
    random_state: int or RandomState, optional
        random seed

    Note
    ----
    Setting the ``random_state`` may be important for getting consistent
    looking predictions when many chunks/subchunks are used. This is because
    the predictive distribution is sampled for these algorithms!
    """

    def __init__(self, kern='rbf', nbases=50, lenscale=1., var=1., falloff=1.,
                 regulariser=1., ard=True, indicator_field='censored',
                 maxiter=3000, batch_size=10, alpha=0.01, beta1=0.9,
                 beta2=0.99, epsilon=1e-8, random_state=None):

        lhood = Switching(lenscale=falloff,
                          var_init=Parameter(var, Positive()))

        super().__init__(likelihood=lhood,
                         basis=None,
                         regulariser=Parameter(regulariser, Positive()),
                         maxiter=maxiter,
                         batch_size=batch_size,
                         updater=Adam(alpha, beta1, beta2, epsilon),
                         random_state=random_state
                         )

        self.indicator_field = indicator_field
        self._store_params(kern, nbases, lenscale, ard)

    def fit(self, X, y, fields):
        r"""
        Learn the parameters of the approximate Gaussian process.

        Parameters
        ----------
        X: ndarray
            (N, d) array input dataset (N samples, d dimensions).
        y: ndarray
            (N,) array targets (N samples)
        fields: dict
            dictionary of fields parsed from the shape file.
            ``indicator_field`` should be a key in this dictionary. This should
            be of the form:

            .. code::

                fields = {'censored': ['yes', 'no', ..., 'yes'],
                          'otherfield': ...,
                          ...
                          }

            where ``censored`` is the indicator field, and ``yes`` means we
            have not directly observed the phenomena, and ``no`` means we have.
        """

        largs = self._parse_largs(fields[self.indicator_field])
        return super().fit(X, y, likelihood_args=(largs,))

    def predict_proba(self, X, interval=0.95, fields={}):
        """
        Predictive mean and variance from an approximate Gaussian process.

        Parameters
        ----------
        X: ndarray
            (Ns, d) array query dataset (Ns samples, d dimensions).
        interval: float, optional
            The percentile confidence interval (e.g. 95%) to return.
        fields: dict, optional
            dictionary of fields parsed from the shape file.
            ``indicator_field`` should be a key in this dictionary. If this is
            not present, then a Gaussian likelihood will be used for all
            predictions. The only time this may be input if for cross
            validation.

        Returns
        -------
        Ey: ndarray
            The expected value of ys for the query inputs, X of shape (Ns,).
        Vy: ndarray
            The expected variance of ys (excluding likelihood noise terms) for
            the query inputs, X of shape (Ns,).
        ql: ndarray
            The lower end point of the interval with shape (Ns,)
        qu: ndarray
            The upper end point of the interval with shape (Ns,)
        """

        if self.indicator_field in fields:
            largs = self._parse_largs(fields[self.indicator_field])
        else:
            largs = np.ones(len(X), dtype=bool)

        return super().predict_proba(X, interval, likelihood_args=(largs,))

    def _parse_largs(self, largs):

        return np.array([v == 'no' for v in largs], dtype=bool)


#
# Approximate probabilistic output for Random Forest
#

class RandomForestRegressor(RFR):
    """
    Implements a "probabilistic" output by looking at the variance of the
    decision tree estimator ouputs.
    """

    def predict_proba(self, X, interval=0.95):
        Ey = self.predict(X)

        Vy = np.zeros_like(Ey)
        for dt in self.estimators_:
            Vy += (dt.predict(X) - Ey)**2

        Vy /= len(self.estimators_)

        # FIXME what if elements of Vy are zero?

        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


#
# Target Transformer factory
#

def transform_targets(Learner):

    class TransformedLearner(Learner):
        # NOTE: All of these explicitly ignore **kwargs on purpose. All generic
        # revrand and scikit learn algorithms don't need them. Custom models
        # probably shouldn't be using this factory

        def __init__(self, target_transform='identity', *args, **kwargs):

            super().__init__(*args, **kwargs)
            self.ytform = transforms.transforms[target_transform]()

        def fit(self, X, y, *args, **kwargs):

            self.ytform.fit(y)
            y_t = self.ytform.transform(y)

            return super().fit(X, y_t)

        def predict(self, X, *args, **kwargs):

            Ey_t = super().predict(X)
            Ey = self.ytform.itransform(Ey_t)

            return Ey

        if hasattr(Learner, 'predict_proba'):
            def predict_proba(self, X, interval=0.95, *args, **kwargs):

                Ns = X.shape[0]
                nsamples = 100

                # Expectation and variance in latent space
                Ey_t, Sy_t, ql, qu = super().predict_proba(X, interval)

                Sy_t = np.sqrt(Sy_t)  # inplace to save mem

                # Now transform expectation, and sample to get transformed
                # variance
                Ey = self.ytform.itransform(Ey_t)
                ql = self.ytform.itransform(ql)
                qu = self.ytform.itransform(qu)
                Vy = np.zeros_like(Ey)

                # Do this as much in place as possible for memory conservation
                for _ in range(nsamples):
                    ys = self.ytform.itransform(Ey_t + np.random.randn(Ns)
                                                * Sy_t)
                    Vy += (Ey - ys)**2

                Vy /= nsamples

                return Ey, Vy, ql, qu

    return TransformedLearner


#
# Construct compatible classes for the pipeline, these need to be module level
# for pickling...
#

class SVRTransformed(transform_targets(SVR), TagsMixin):
    pass


class LinearRegTransformed(transform_targets(LinearReg), TagsMixin):
    pass


class RandomForestTransformed(transform_targets(RandomForestRegressor),
                              TagsMixin):
    pass


class ApproxGPTransformed(transform_targets(ApproxGP), TagsMixin):
    pass


class ARDRegressionTransformed(transform_targets(ARDRegression), TagsMixin):
    pass


class DecisionTreeTransformed(transform_targets(DecisionTreeRegressor),
                              TagsMixin):
    pass


class ExtraTreeTransformed(transform_targets(ExtraTreeRegressor), TagsMixin):
    pass


class SGDLinearRegTransformed(transform_targets(SGDLinearReg), TagsMixin):
    pass


class SGDApproxGPTransformed(transform_targets(SGDApproxGP), TagsMixin):
    pass


class CubistTransformed(transform_targets(Cubist), TagsMixin):
    pass


#
# Helper functions for multiple outputs and missing/masked data
#

def apply_masked(func, data, args=(), kwargs={}):
    # Data is just a matrix (i.e. X for prediction)

    # No masked data
    if np.ma.count_masked(data) == 0:
        return func(data.data, *args, **kwargs)

    # Prediction with missing inputs
    okdata = (data.mask.sum(axis=1)) == 0 if data.ndim == 2 else ~data.mask
    res = func(data.data[okdata], *args, **kwargs)

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


def apply_multiple_masked(func, data, args=(), kwargs={}):
    # Data is a sequence of arrays (i.e. X, y pairs for training)

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

    unstackfunc = lambda catdata, *nargs, **nkwargs: \
        func(*chain(unstack(catdata), nargs), **nkwargs)

    return apply_masked(unstackfunc, np.ma.hstack(datastack), args, kwargs)


#
# Static module properties
#

modelmaps = {'randomforest': RandomForestTransformed,
             'bayesreg': LinearRegTransformed,
             'sgdbayesreg': SGDLinearRegTransformed,
             'approxgp': ApproxGPTransformed,
             'sgdapproxgp': SGDApproxGPTransformed,
             'svr': SVRTransformed,
             'ardregression': ARDRegressionTransformed,
             'decisiontree': DecisionTreeTransformed,
             'extratree': ExtraTreeTransformed,
             'cubist': CubistTransformed,
             'depthregress': DepthRegressor,
             }


basismap = {'rbf': RandomRBF,
            'laplace': RandomLaplace,
            'cauchy': RandomCauchy,
            'matern32': RandomMatern32,
            'matern52': RandomMatern52
            }
