"""
Model Objects and ML algorithm serialisation.

This module makes many of the models in `scikit learn
<http://scikit-learn.org/>`_ and 
`revrand <https://github.com/NICTA/revrand>`_ available to our 
pipeline, as well as augmenting their functionality with, for examples, 
target transformations.

This table is a quick breakdown of the advantages and disadvantages of 
the various algorithms we can use in this pipeline.

==========================  ====================  ==================  ================  =============
Algorithm                   Learning Scalability  Modelling Capacity  Prediction Speed  Probabilistic
==========================  ====================  ==================  ================  =============
Bayesian linear regression  \+ \+ \+              \+                  \+ \+ \+ \+       Yes
Approx. Gaussian process    \+ \+                 \+ \+ \+ \+         \+ \+ \+ \+       Yes
SGD linear regression       \+ \+ \+ \+           \+                  \+ \+ \+          Yes
SGD Gaussian process        \+ \+ \+ \+           \+ \+ \+ \+         \+ \+ \+          Yes
Support Vector Regression   \+                    \+ \+ \+ \+         \+                No
Random Forest Regression    \+ \+ \+              \+ \+ \+ \+         \+ \+             Pseudo
Cubist Regression           \+ \+ \+              \+ \+ \+ \+         \+ \+             Pseudo
ARD Regression              \+ \+                 \+ \+               \+ \+ \+          No
Extremely Randomized Reg.   \+ \+ \+              \+ \+ \+ \+         \+ \+             No
Decision Tree Regression    \+ \+ \+              \+ \+ \+            \+ \+ \+ \+       No
==========================  ====================  ==================  ================  =============
"""
import inspect
import os
import pickle
import warnings
import logging
from itertools import chain
from os.path import join, isdir, abspath
import numpy as np
from revrand import StandardLinearModel, GeneralisedLinearModel
from revrand.basis_functions import LinearBasis, RandomRBF, \
    RandomLaplace, RandomCauchy, RandomMatern32, RandomMatern52
from revrand.btypes import Parameter, Positive
from revrand.likelihoods import Gaussian
from revrand.optimize import Adam
from revrand.utils import atleast_list
from scipy.integrate import fixed_quad
from scipy.stats import norm

from sklearn.svm import SVR, SVC
from sklearn.ensemble import (RandomForestRegressor as RFR,
                              RandomForestClassifier as RFC,
                              GradientBoostingClassifier)
from sklearn.linear_model import ARDRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.kernel_approximation import RBFSampler

from uncoverml import mpiops
from uncoverml.resampling import bootstrap_data_indicies
from uncoverml.interpolate import SKLearnNearestNDInterpolator, \
    SKLearnLinearNDInterpolator, SKLearnRbf, SKLearnCT
from uncoverml.cubist import Cubist
from uncoverml.cubist import MultiCubist
from uncoverml.transforms import target as transforms
warnings.filterwarnings("ignore", category=DeprecationWarning)
#
# Module constants
#

_logger = logging.getLogger(__name__)
QUADORDER = 5  # Order of quadrature used for transforming probabilistic vals


#
# Mixin classes for providing pipeline compatibility to revrand
#

class BasisMakerMixin():
    """
    Mixin class for easily creating approximate kernel functions for revrand.

    This is primarily used for the approximate Gaussian process algorithms.
    """

    def fit(self, X, y, *args, **kwargs):

        self._make_basis(X)
        return super().fit(X, y, *args, **kwargs)  # args for GLMs

    def _store_params(self, kernel, regulariser, nbases, lenscale, ard):

        self.kernel = kernel
        self.nbases = nbases
        self.ard = ard
        self.lenscale = lenscale if np.isscalar(lenscale) \
            else np.asarray(lenscale)
        self.regulariser = Parameter(regulariser, Positive())

    def _make_basis(self, X):

        D = X.shape[1]
        lenscale = self.lenscale
        if self.ard and D > 1:
            lenscale = np.ones(D) * lenscale
        lenscale_init = Parameter(lenscale, Positive())
        gpbasis = basismap[self.kernel](Xdim=X.shape[1], nbases=self.nbases,
                                        lenscale=lenscale_init,
                                        regularizer=self.regulariser)

        self.basis = gpbasis + LinearBasis()


class PredictDistMixin():
    """
    Mixin class for providing a ``predict_dist`` method to the
    StandardLinearModel class in revrand.
    """

    def predict_dist(self, X, interval=0.95, *args, **kwargs):
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


class GLMPredictDistMixin():
    """
    Mixin class for providing a ``predict_dist`` method to the
    GeneralisedLinearModel class in revrand.

    This is especially for use with Gaussian likelihood models.
    """

    def predict_dist(self, X, interval=0.95, *args, **kwargs):
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
        Vy += self.like_hypers_
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


class MutualInfoMixin():
    """
    Mixin class for providing predictive entropy reduction functionality to the
    StandardLinearModel class (only).
    """

    def entropy_reduction(self, X):
        """
        Predictice entropy reduction (a.k.a mutual information).

        Estimate the reduction in the posterior distribution's entropy (i.e.
        model uncertainty reduction) as a result of including a particular
        observation.

        Parameters
        ----------
        X: ndarray
            (Ns, d) array query dataset (Ns samples, d dimensions).

        Returns
        -------
        MI: ndarray
            Prediction of mutual information (expected reduiction in posterior
            entrpy) assocated with each query input. The units are 'nats', and
            the shape of the returned array is (Ns,).
        """
        Phi = self.basis.transform(X, *atleast_list(self.hypers_))
        pCp = [p.dot(self.covariance_).dot(p.T) for p in Phi]
        MI = 0.5 * (np.log(self.var_ + np.array(pCp)) - np.log(self.var_))
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
            ``predict``, `predict_dist``, ``entropy_reduction``).
        """

        # Classification
        if hasattr(self, 'predict_proba'):
            tags = self.get_classes()
            return tags

        # Regression
        tags = ['Prediction']
        if hasattr(self, 'predict_dist'):
            tags.extend(['Variance', 'Lower quantile', 'Upper quantile'])

        if hasattr(self, 'entropy_reduction'):
            tags.append('Expected reduction in entropy')

        if hasattr(self, 'krige_residual'):
            tags.append('Kriged correction')

        if hasattr(self, 'ml_prediction'):
            tags.append('ml prediction')

        return tags


#
# Specialisation of revrand's interface to work from the command line with a
# few curated algorithms
#

class LinearReg(StandardLinearModel, PredictDistMixin, MutualInfoMixin):
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
    nstarts : int, optional
        if there are any parameters with distributions as initial values, this
        determines how many random candidate starts shoulds be evaluated before
        commencing optimisation at the best candidate.
    """

    def __init__(self, onescol=True, var=1., regulariser=1., tol=1e-8,
                 maxiter=1000, nstarts=100):

        basis = LinearBasis(onescol=onescol,
                            regularizer=Parameter(regulariser, Positive()))
        super().__init__(basis=basis,
                         var=Parameter(var, Positive()),
                         tol=tol,
                         maxiter=maxiter,
                         nstarts=nstarts
                         )


class ApproxGP(BasisMakerMixin, StandardLinearModel, PredictDistMixin,
               MutualInfoMixin):
    """
    An approximate Gaussian process for medium scale data.

    Parameters
    ----------
    kernel: str, optional
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
    nstarts : int, optional
        if there are any parameters with distributions as initial values, this
        determines how many random candidate starts shoulds be evaluated before
        commencing optimisation at the best candidate.
    """

    def __init__(self, kernel='rbf', nbases=50, lenscale=1., var=1.,
                 regulariser=1., ard=True, tol=1e-8, maxiter=1000,
                 nstarts=100):

        super().__init__(basis=None,
                         var=Parameter(var, Positive()),
                         tol=tol,
                         maxiter=maxiter,
                         nstarts=nstarts
                         )

        self._store_params(kernel, regulariser, nbases, lenscale, ard)


class SGDLinearReg(GeneralisedLinearModel, GLMPredictDistMixin):
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
    nstarts : int, optional
        if there are any parameters with distributions as initial values, this
        determines how many random candidate starts shoulds be evaluated before
        commencing optimisation at the best candidate.
    Note
    ----
    Setting the ``random_state`` may be important for getting consistent
    looking predictions when many chunks/subchunks are used. This is because
    the predictive distribution is sampled for these algorithms!
    """

    def __init__(self, onescol=True, var=1., regulariser=1., maxiter=3000,
                 batch_size=10, alpha=0.01, beta1=0.9, beta2=0.99,
                 epsilon=1e-8, random_state=None, nstarts=500):
        basis = LinearBasis(onescol=onescol,
                            regularizer=Parameter(regulariser, Positive()))
        super().__init__(likelihood=Gaussian(Parameter(var, Positive())),
                         basis=basis,
                         maxiter=maxiter,
                         batch_size=batch_size,
                         updater=Adam(alpha, beta1, beta2, epsilon),
                         random_state=random_state,
                         nstarts=nstarts
                         )


class SGDApproxGP(BasisMakerMixin, GeneralisedLinearModel,
                  GLMPredictDistMixin):
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
    var: float, optional
        observation variance initial value.
    regulariser: float, optional
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
    nstarts : int, optional
        if there are any parameters with distributions as initial values, this
        determines how many random candidate starts shoulds be evaluated before
        commencing optimisation at the best candidate.
    Note
    ----
    Setting the ``random_state`` may be important for getting consistent
    looking predictions when many chunks/subchunks are used. This is because
    the predictive distribution is sampled for these algorithms!
    """

    def __init__(self, kernel='rbf', nbases=50, lenscale=1., var=1.,
                 regulariser=1., ard=True, maxiter=3000, batch_size=10,
                 alpha=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8,
                 random_state=1, nstarts=500):

        super().__init__(likelihood=Gaussian(Parameter(var, Positive())),
                         basis=None,
                         maxiter=maxiter,
                         batch_size=batch_size,
                         updater=Adam(alpha, beta1, beta2, epsilon),
                         random_state=random_state,
                         nstarts=nstarts
                         )
        self._store_params(kernel, regulariser, nbases, lenscale, ard)

#
# Approximate probabilistic output for Random Forest
#


class RandomForestRegressor(RFR):
    """
    Implements a "probabilistic" output by looking at the variance of the
    decision tree estimator ouputs.
    """

    def predict_dist(self, X, interval=0.95):
        if hasattr(self, "_notransform_predict"):
            Ey = self._notransform_predict(X)
        else:
            Ey = self.predict(X)

        Vy = np.zeros_like(Ey)
        for dt in self.estimators_:
            Vy += (dt.predict(X) - Ey)**2

        Vy /= len(self.estimators_)

        # FIXME what if elements of Vy are zero?

        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


class RandomForestRegressorMulti():

    def __init__(self,
                 outdir='.',
                 forests=10,
                 parallel=True,
                 n_estimators=10,
                 random_state=1,
                 **kwargs):
        self.forests = forests
        self.n_estimators = n_estimators
        self.parallel = parallel
        self.kwargs = kwargs
        self.random_state = random_state
        self._trained = False
        self._randomforests = {}

    def fit(self, x, y, *args, **kwargs):
        if self.parallel:
            process_rfs = np.array_split(range(self.forests),
                                         mpiops.chunks)[mpiops.chunk_index]
        else:
            process_rfs = range(self.forests)

        for t in process_rfs:
            _logger.info(':mpi:training forest {} using process {}'.format(t, mpiops.chunk_index))

            np.random.seed(self.random_state + t)

            # change random state in each forest
            self.kwargs['random_state'] = np.random.randint(0, 10000)
            rf = RandomForestTransformed(
                n_estimators=self.n_estimators, **self.kwargs)
            rf.fit(x, y)
            if self.parallel:  # used in training
                self._randomforests['rf_model_{}'.format(t)] = rf
            else:  # used when parallel is false, i.e., during x-val
                self._randomforests['rf_model_{}_{}'.format(t, mpiops.chunk_index)] = rf
        if self.parallel:
            mpiops.comm.barrier()
        # Mark that we are now trained
        self._trained = True

    def predict_dist(self, x, interval=0.95, *args, **kwargs):
        # We can't make predictions until we have trained the model
        if not self._trained:
            _logger.warning(':mpi:Train first')
            return

        y_pred = np.zeros((x.shape[0], self.forests * self.n_estimators))

        for t in range(self.forests):
            if self.parallel:  # used in training
                f = self._randomforests['rf_model_{}'.format(t)]
            else:  # used when parallel is false, i.e., during x-val
                f = self._randomforests['rf_model_{}_{}'.format(t, mpiops.chunk_index)]
            for m, dt in enumerate(f.estimators_):
                y_pred[:, t * self.n_estimators + m] = dt.predict(x)

        y_mean = np.mean(y_pred, axis=1)
        y_var = np.var(y_pred, axis=1)

        # Determine quantiles
        ql, qu = norm.interval(interval, loc=y_mean, scale=np.sqrt(y_var))

        return y_mean, y_var, ql, qu

    def predict(self, x):
        return self.predict_dist(x)[0]

#
# Approximate large scale kernel classifier factory
#


def kernelize(classifier):

    class ClassifierRBF:

        def __init__(self, gamma='auto', n_components=100, random_state=None,
                     **kwargs):
            self.gamma = gamma
            self.n_components = n_components
            self.random_state = random_state
            self.clf = classifier(**kwargs)

        def fit(self, X, y):
            if self.gamma == 'auto':
                D = X.shape[1]
                self.gamma = 1 / D
            self.rbf = RBFSampler(
                gamma=self.gamma,
                n_components=self.n_components,
                random_state=self.random_state
            )

            self.clf.fit(self.rbf.fit_transform(X), y)
            return self

        def predict(self, X):
            p = self.clf.predict(self.rbf.transform(X))
            return p

        def predict_proba(self, X):
            p = self.clf.predict_proba(self.rbf.transform(X))
            return p

    return ClassifierRBF

#
# Bootstrap Model Factory
#

def bootstrap_model(model):
    class BootstrappedModel():
        def __init__(self, n_models=100, parallel=True, *args, **kwargs):
            # 'bootstrap' attr is dinky workaround for checking if a 
            # model is a bootstrap model (can't get at BootstrappedModel
            # class as it's in scope of factory function).
            self.__bootstrapped_model__ = 'bootstrap'
            self.n_models = n_models
            self.models = [model(*args, **kwargs) for i in range(self.n_models)]
            self.parallel = parallel

        def fit(self, X, y, lon_lat, *args, **kwargs):
            _logger.info('Training %s bootstrapped models', self.n_models)

            if self.parallel:
                models = np.array_split(self.models, mpiops.chunks)[mpiops.chunk_index]
            else:
                models = self.models

            for i, m in enumerate(models):
                inds = bootstrap_data_indicies(len(y), random_state=mpiops.chunk_index + 1 + i)
                bsx = X[inds]
                bsy = y[inds]
                m.fit(bsx, bsy)
                _logger.info(':mpi:Trained model %s of %s', i + 1, len(models))

            if self.parallel:
                models = mpiops.comm.gather(models, root=0)
                if mpiops.chunk_index == 0:
                    self.models = list(chain.from_iterable(models))


        def predict_dist(self, X, interval=0.95, bootstrap_predictions=None, *args, **kwargs):
            n_predictions = bootstrap_predictions if bootstrap_predictions is not None \
                else len(self.models)
            model_chunks = self.models[:n_predictions]
            predictions = []
            for i, m in enumerate(model_chunks):
                _logger.info(f":mpi:Predicting bootstrapped model %s of %s", 
                    i + 1, len(model_chunks))
                predictions.append(m.predict(X, *args, **kwargs))

            predictions = np.array(predictions)
            y_mean = np.ma.mean(predictions, axis=0)
            y_var = np.ma.var(predictions, axis=0)
            ql, qu = norm.interval(interval, loc=y_mean, scale=np.sqrt(y_var))
            return y_mean, y_var, ql, qu

        def predict(self, X, bootstrap_predictions=None):
            return self.predict_dist(X, bootstrap_predictions=bootstrap_predictions)[0]

    return BootstrappedModel

#
# Target Transformer factory
#


def transform_targets(Regressor):
    """
    Factory function that add's target transformation capabiltiy to compatible
    scikit learn objects.

    Look at the ``transformers.py`` module for more information on valid target
    transformers.

    Example
    -------
    >>> svr = transform_targets(SVR)(target_transform='Standardise', gamma=0.1)

    """
    class TransformedRegressor(Regressor):
        # NOTE: All of these explicitly ignore **kwargs on purpose. All generic
        # revrand and scikit learn algorithms don't need them. Custom models
        # probably shouldn't be using this factory

        def __init__(self, target_transform='identity', *args, **kwargs):

            super().__init__(*args, **kwargs)
            self.target_transform = transforms.transforms[target_transform]()

        def fit(self, X, y, *args, **kwargs):

            self.target_transform.fit(y)
            y_t = self.target_transform.transform(y)
            # Hack to check if we can apply sample weights
            if 'sample_weight' in inspect.signature(super().fit).parameters.keys() \
                    and 'sample_weight' in kwargs.keys():
                return super().fit(X, y_t, sample_weight=kwargs['sample_weight'])
            else:
                return super().fit(X, y_t)

        def _notransform_predict(self, X, *args, **kwargs):
            Ey = super().predict(X)
            return Ey

        def predict(self, X, *args, **kwargs):

            Ey_t = self._notransform_predict(X, *args, **kwargs)
            Ey = self.target_transform.itransform(Ey_t)
            return Ey

        if hasattr(Regressor, 'predict_dist'):
            def predict_dist(self, X, interval=0.95, *args, **kwargs):

                # Expectation and variance in latent space
                Ey_t, Vy_t, ql, qu = super().predict_dist(X, interval)

                # Save computation if identity transform
                if type(self.target_transform) is transforms.Identity:
                    return Ey_t, Vy_t, ql, qu

                # Save computation if standardise transform
                elif type(self.target_transform) is transforms.Standardise:
                    Ey = self.target_transform.itransform(Ey_t)
                    Vy = Vy_t * self.target_transform.ystd ** 2
                    ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))
                    return Ey, Vy, ql, qu

                # All other transforms require quadrature
                Ey = np.empty_like(Ey_t)
                Vy = np.empty_like(Vy_t)

                # Used fixed order quadrature to transform prob. estimates
                for i, (Eyi, Vyi) in enumerate(zip(Ey_t, Vy_t)):

                    # Establish bounds
                    Syi = np.sqrt(Vyi)
                    a, b = Eyi - 3 * Syi, Eyi + 3 * Syi  # approx 99% bounds

                    # Quadrature
                    Ey[i], _ = fixed_quad(self.__expec_int, a, b, n=QUADORDER,
                                          args=(Eyi, Syi))
                    Vy[i], _ = fixed_quad(self.__var_int, a, b, n=QUADORDER,
                                          args=(Ey[i], Eyi, Syi))

                ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

                return Ey, Vy, ql, qu

        def __expec_int(self, x, mu, std):
            px = _normpdf(x, mu, std)
            Ex = self.target_transform.itransform(x) * px
            return Ex

        def __var_int(self, x, Ex, mu, std):
            px = _normpdf(x, mu, std)
            Vx = (self.target_transform.itransform(x) - Ex) ** 2 * px
            return Vx

    return TransformedRegressor


#
# Label Encoder Factory
#

def encode_targets(Classifier):

    class EncodedClassifier(Classifier):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.le = LabelEncoder()
            self.prefitted = False

        def prefit_encoder(self, y):
            self.prefitted = True
            return self.le.fit_transform(y)

        def fit(self, X, y, **kwargs):
            if self.prefitted:
                y_t = self.le.transform(y)
            else:
                y_t = self.le.fit_transform(y)
            return super().fit(X, y_t)

        def predict_proba(self, X, **kwargs):
            p = super().predict_proba(X)
            y = np.argmax(p, axis=1)  # Also return hard labels
            return y, p

        def get_classes(self):
            tags = ["most_likely"]
            tags += ["{}_{}".format(c, i)
                     for i, c in enumerate(self.le.classes_)]
            return tags

    return EncodedClassifier

#
# Construct compatible classes for the pipeline, these need to be module level
# for pickling...
#


class CustomKNeighborsRegressor(KNeighborsRegressor):

    def __init__(self, n_neighbors=10,  # min_weight_fraction,
                 weights='distance',
                 algorithm='auto',
                 leaf_size=30,
                 metric='minkowski', p=2,
                 metric_params=None, n_jobs=1,
                 min_distance=0.0):

        self.min_distance = min_distance

        weights_ = weights

        if weights == 'distance':
            weights_ = self._get_weights

        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights_,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params, n_jobs=n_jobs
        )

    def _get_weights(self, dist):
        # if user attempts to classify a point that was zero distance from one
        # or more training points, those training points are weighted as 1.0
        # and the other points as 0.0
        if dist.dtype is np.dtype(object):
            for point_dist_i, point_dist in enumerate(dist):
                # check if point_dist is iterable
                # (ex: RadiusNeighborClassifier.predict may set an element of
                # dist to 1e-6 to represent an 'outlier')
                if hasattr(point_dist, '__contains__') and 0. in point_dist:
                    dist[point_dist_i] = point_dist == 0. + self.min_distance
                else:
                    dist[point_dist_i] = 1. / (point_dist + self.min_distance)
        else:
            dist = 1. / (dist + self.min_distance)

        return dist


# Define transformed models

class KNearestNeighborTransformed(transform_targets(CustomKNeighborsRegressor),
                                  TagsMixin):
    """
    K Nearest Neighbour Regression

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    """

    pass


class SVRTransformed(transform_targets(SVR), TagsMixin):
    """
    Support vector machine.

    http://scikit-learn.org/dev/modules/svm.html#svm
    """
    pass


class LinearRegTransformed(transform_targets(LinearReg), TagsMixin):
    """
    Bayesian linear regression.

    http://nicta.github.io/revrand/slm.html
    """
    pass


class RandomForestTransformed(transform_targets(RandomForestRegressor),
                              TagsMixin):
    """
    Random forest regression.

    http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    """
    pass


class MultiRandomForestTransformed(
        transform_targets(RandomForestRegressorMulti), TagsMixin):
    """
    MPI implementation of Random forest regression with forest grown on
    many CPUS.

    http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    """
    pass


class ApproxGPTransformed(transform_targets(ApproxGP), TagsMixin):
    """
    Approximate Gaussian process.

    http://nicta.github.io/revrand/slm.html
    """
    pass


class ARDRegressionTransformed(transform_targets(ARDRegression), TagsMixin):
    """
    ARD regression.

    http://scikit-learn.org/dev/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression
    """
    pass


class DecisionTreeTransformed(transform_targets(DecisionTreeRegressor),
                              TagsMixin):
    """
    Decision tree regression.

    http://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
    """
    pass


class ExtraTreeTransformed(transform_targets(ExtraTreeRegressor), TagsMixin):
    """
    Extremely randomised tree regressor.

    http://scikit-learn.org/dev/modules/generated/sklearn.tree.ExtraTreeRegressor.html#sklearn.tree.ExtraTreeRegressor
    """
    pass


class SGDLinearRegTransformed(transform_targets(SGDLinearReg), TagsMixin):
    """
    Baysian linear regression with stochastic gradients.

    http://nicta.github.io/revrand/glm.html
    """
    pass


class SGDApproxGPTransformed(transform_targets(SGDApproxGP), TagsMixin):
    """
    Approximate Gaussian processes with stochastic gradients.

    http://nicta.github.io/revrand/glm.html
    """
    pass


class CubistTransformed(transform_targets(Cubist), TagsMixin):
    """
    Cubist regression (wrapper).

    https://www.rulequest.com/cubist-info.html
    """
    pass


class CubistMultiTransformed(transform_targets(MultiCubist), TagsMixin):
    """
    Parallel Cubist regression (wrapper).

    https://www.rulequest.com/cubist-info.html
    """
    pass


class LogisticClassifier(encode_targets(LogisticRegression), TagsMixin):
    """
    Logistic Regression for muli-class classification.

    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    pass


class RandomForestClassifier(encode_targets(RFC), TagsMixin):
    """
    Random Forest for muli-class classification.

    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    pass


class SupportVectorClassifier(encode_targets(SVC), TagsMixin):
    """
    Support Vector Machine multi-class classification.

    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    pass


class GradBoostedTrees(encode_targets(GradientBoostingClassifier), TagsMixin):
    """
    Gradient Boosted Trees multi-class classification.

    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    """
    pass


class LogisticRBF(encode_targets(kernelize(LogisticRegression)), TagsMixin):
    """Approximate large scale kernel logistic regression."""
    pass


class TransformedLinearNDInterpolator(
    transform_targets(SKLearnLinearNDInterpolator), TagsMixin):
    pass


class TransformedNearestNDInterpolator(
    transform_targets(SKLearnNearestNDInterpolator), TagsMixin):
    pass


class TransformedRbfInterpolator(transform_targets(SKLearnRbf), TagsMixin):
    pass


class TransformedCTInterpolator(transform_targets(SKLearnCT), TagsMixin):
    pass

class BootstrappedSVR(bootstrap_model(SVRTransformed), TagsMixin):
    pass


class MaskRows:

    def __init__(self, *Xs):
        self.okrows = self.get_complete_rows(Xs[0])
        if len(Xs) > 1:
            for c in chain(Xs[1:]):
                self.okrows = np.logical_and(
                    self.okrows, self.get_complete_rows(c))

    def trim_mask(self, X):
        if np.ma.isMaskedArray(X):
            predict_data = X.data[self.okrows]

            if predict_data.shape[0] == 0:  # if all of this chunk is masked
                # to get dimension of the func return, we create a dummpy res
                predict_data = np.ones((1, X.data.shape[1]))

            return predict_data
        else:
            return X[self.okrows]

    def trim_masks(self, *Xs):
        return [self.trim_mask(x) for x in Xs]

    def apply_mask(self, X):
        N = len(self.okrows)
        if X.ndim == 2:
            D = X.shape[1]
            Xdat = np.zeros((N, D))
            Xmask = np.zeros((N, D), dtype=bool)
        elif X.ndim == 1:
            Xdat = np.zeros(N)
            Xmask = np.zeros(N, dtype=bool)
        else:
            raise ValueError("Can only mask/unmask 2D arrays")
        Xmask[~self.okrows] = True
        Xdat[self.okrows] = X
        return np.ma.masked_array(data=Xdat, mask=Xmask)

    def apply_masks(self, *Xs):
        return [self.apply_mask(x) for x in Xs]

    @staticmethod
    def get_complete_rows(X):
        N = len(X)
        if np.ma.isMaskedArray(X):
            if np.isscalar(X.mask):
                okrows = ~X.mask * np.ones(N, dtype=bool)
            else:
                okrows = np.all(~X.mask, axis=1) if X.ndim == 2 else ~X.mask
        else:
            okrows = np.ones(N, dtype=bool)
        return okrows


def apply_masked(func, data, *args, **kwargs):
    # Data is just a matrix (i.e. X for prediction)
    mr = MaskRows(data)
    res = func(mr.trim_mask(data), *args, **kwargs)

    # For training/fitting that returns nothing
    if not isinstance(res, np.ndarray):
        return res
    else:
        return mr.apply_mask(res)


def apply_multiple_masked(func, data, *args, **kwargs):
    # Data is a sequence of arrays (i.e. X, y pairs for training)
    mr = MaskRows(*data)
    res = func(*chain(mr.trim_masks(*data), args), **kwargs)

    # For training/fitting that returns nothing
    if not isinstance(res, np.ndarray):
        return res
    else:
        return mr.apply_mask(res)
#
# Static module properties
#

# Add all models available to the learning pipeline here!
bootstrapped = {
    'bootstrapsvr': BootstrappedSVR
}

regressors = {
    'randomforest': RandomForestTransformed,
    'multirandomforest': MultiRandomForestTransformed,
    'bayesreg': LinearRegTransformed,
    'sgdbayesreg': SGDLinearRegTransformed,
    'approxgp': ApproxGPTransformed,
    'sgdapproxgp': SGDApproxGPTransformed,
    'svr': SVRTransformed,
    'ardregression': ARDRegressionTransformed,
    'decisiontree': DecisionTreeTransformed,
    'extratree': ExtraTreeTransformed,
    'cubist': CubistTransformed,
    'multicubist': CubistMultiTransformed,
    'nnr': KNearestNeighborTransformed,
}


interpolators = {
    'linear': TransformedLinearNDInterpolator,
    'nn': TransformedNearestNDInterpolator,
    'rbf': TransformedRbfInterpolator,
    'cubic2d': TransformedCTInterpolator,
}


classifiers = {
    'logistic': LogisticClassifier,
    'logisticrbf': LogisticRBF,
    'forestclassifier': RandomForestClassifier,
    'svc': SupportVectorClassifier,
    'boostedtrees': GradBoostedTrees
}

modelmaps = {**classifiers, **regressors, **interpolators, **bootstrapped}

# Add all kernels for the approximate Gaussian processes here!
basismap = {
    'rbf': RandomRBF,
    'laplace': RandomLaplace,
    'cauchy': RandomCauchy,
    'matern32': RandomMatern32,
    'matern52': RandomMatern52
}

#
# Private functions and constants
#


_SQRT2PI = np.sqrt(2 * np.pi)


# Faster than calling scipy's norm.pdf for quadrature. This is called with HIGH
# frequency!
def _normpdf(x, mu, std):
    if std == 0:
        _logger.warning("STD of 0 in _normpdf, stabilising with very small value")
        std += np.nextafter(np.float32(0), np.float32(1))
    return 1. / (_SQRT2PI * std) * np.exp(-0.5 * ((x - mu) / std)**2)
