"""Model Objects and ML algorithm serialisation."""

import os
import pickle
from itertools import chain
from os.path import join, isdir, abspath

import numpy as np
from revrand import StandardLinearModel, GeneralisedLinearModel
from revrand.basis_functions import (LinearBasis, RandomRBF, RandomLaplace,
                                     RandomCauchy, RandomMatern32,
                                     RandomMatern52)
from revrand.btypes import Parameter, Positive
from revrand.likelihoods import Gaussian
from revrand.optimize import Adam
from revrand.utils import atleast_list
from scipy.integrate import fixed_quad
from scipy.stats import norm
from sklearn.ensemble import (RandomForestRegressor as RFR,
                              RandomForestClassifier as RFC,
                              GradientBoostingClassifier)
from sklearn.linear_model import ARDRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.preprocessing import LabelEncoder
from uncoverml import mpiops
from uncoverml.cubist import Cubist
from uncoverml.cubist import MultiCubist
from uncoverml.likelihoods import Switching
from uncoverml.transforms import target as transforms

#
# Module constants
#

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
                 random_state=None, nstarts=500):

        super().__init__(likelihood=Gaussian(Parameter(var, Positive())),
                         basis=None,
                         maxiter=maxiter,
                         batch_size=batch_size,
                         updater=Adam(alpha, beta1, beta2, epsilon),
                         random_state=random_state,
                         nstarts=nstarts
                         )
        self._store_params(kernel, regulariser, nbases, lenscale, ard)


# Bespoke regressor for basin-depth problems
class DepthRegressor(BasisMakerMixin, GeneralisedLinearModel, TagsMixin,
                     GLMPredictDistMixin):
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

    def __init__(self, kernel='rbf', nbases=50, lenscale=1., var=1.,
                 falloff=1., regulariser=1., ard=True,
                 indicator_field='censored', maxiter=3000,
                 batch_size=10, alpha=0.01, beta1=0.9,
                 beta2=0.99, epsilon=1e-8, random_state=None):

        lhood = Switching(lenscale=falloff,
                          var_init=Parameter(var, Positive()))

        super().__init__(likelihood=lhood,
                         basis=None,
                         maxiter=maxiter,
                         batch_size=batch_size,
                         updater=Adam(alpha, beta1, beta2, epsilon),
                         random_state=random_state
                         )

        self.indicator_field = indicator_field
        self._store_params(kernel, regulariser, nbases, lenscale, ard)

    def fit(self, X, y, fields, **kwargs):
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
        # uncomment next line to work with depthregress if your shapefile
        # does not have a 'censored' column
        # fields[self.indicator_field] = ['No'] * len(y)
        largs = self._parse_largs(fields[self.indicator_field])
        return super().fit(X, y, likelihood_args=(largs,))

    def predict_dist(self, X, interval=0.95, fields={}, **kwargs):
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

        return super().predict_dist(X, interval, likelihood_args=(largs,))

    def _parse_largs(self, largs):
        return np.array([v == 'No' for v in largs], dtype=bool)


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


class RandomForestRegressorMulti(TagsMixin):

    def __init__(self,
                 outdir='.',
                 forests=10,
                 parallel=True,
                 n_estimators=10,
                 random_state=1,
                 target_transform='identity',
                 **kwargs):
        self.forests = forests
        self.n_estimators = n_estimators
        self.parallel = parallel
        self.kwargs = kwargs
        self.random_state = random_state
        self._trained = False
        assert isdir(abspath(outdir)), 'Make sure the outdir exists ' \
                                       'and writeable'
        self.temp_dir = join(abspath(outdir), 'results')
        os.makedirs(self.temp_dir, exist_ok=True)
        self.target_transform = target_transform

    def fit(self, x, y, *args, **kwargs):

        # set a different random seed for each thread
        np.random.seed(self.random_state + mpiops.chunk_index)

        if self.parallel:
            process_rfs = np.array_split(range(self.forests),
                                         mpiops.chunks)[mpiops.chunk_index]
        else:
            process_rfs = range(self.forests)

        for t in process_rfs:
            print('training forest {} using '
                  'process {}'.format(t, mpiops.chunk_index))

            # change random state in each forest
            self.kwargs['random_state'] = np.random.randint(0, 10000)
            rf = RandomForestTransformed(
                target_transform=self.target_transform,
                n_estimators=self.n_estimators,
                **self.kwargs
                )
            rf.fit(x, y)
            if self.parallel:  # used in training
                pk_f = join(self.temp_dir,
                            'rf_model_{}.pk'.format(t))
            else:  # used when parallel is false, i.e., during x-val
                pk_f = join(self.temp_dir,
                            'rf_model_{}_{}.pk'.format(t, mpiops.chunk_index))
            with open(pk_f, 'wb') as fp:
                pickle.dump(rf, fp)
        if self.parallel:
            mpiops.comm.barrier()
        # Mark that we are now trained
        self._trained = True

    def predict_dist(self, x, interval=0.95, *args, **kwargs):

        # We can't make predictions until we have trained the model
        if not self._trained:
            print('Train first')
            return

        y_pred = np.zeros((x.shape[0], self.forests * self.n_estimators))

        for i in range(self.forests):
            if self.parallel:  # used in training
                pk_f = join(self.temp_dir,
                            'rf_model_{}.pk'.format(i))
            else:  # used when parallel is false, i.e., during x-val
                pk_f = join(self.temp_dir,
                            'rf_model_{}_{}.pk'.format(i, mpiops.chunk_index))
            with open(pk_f, 'rb') as fp:
                f = pickle.load(fp)
                for m, dt in enumerate(f.estimators_):
                    y_pred[:, i * self.n_estimators + m] = \
                        f.ytform.itransform(dt.predict(x))

        y_mean = np.mean(y_pred, axis=1)
        y_var = np.var(y_pred, axis=1)

        # Determine quantiles
        ql, qu = norm.interval(interval, loc=y_mean, scale=np.sqrt(y_var))

        return y_mean, y_var, ql, qu

    def predict(self, x):
        return self.predict_dist(x)[0]


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
            self.ytform = transforms.transforms[target_transform]()

        def fit(self, X, y, *args, **kwargs):

            self.ytform.fit(y)
            y_t = self.ytform.transform(y)

            return super().fit(X, y_t)

        def _notransform_predict(self, X, *args, **kwargs):
            Ey = super().predict(X)
            return Ey

        def predict(self, X, *args, **kwargs):

            Ey_t = super().predict(X)
            Ey = self.ytform.itransform(Ey_t)

            return Ey

        if hasattr(Regressor, 'predict_dist'):
            def predict_dist(self, X, interval=0.95, *args, **kwargs):

                # Expectation and variance in latent space
                Ey_t, Vy_t, ql, qu = super().predict_dist(X, interval)

                # Save computation if identity transform
                if type(self.ytform) is transforms.Identity:
                    return Ey_t, Vy_t, ql, qu

                # Save computation if standardise transform
                elif type(self.ytform) is transforms.Standardise:
                    Ey = self.ytform.itransform(Ey_t)
                    Vy = Vy_t * self.ytform.ystd**2
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
            Ex = self.ytform.itransform(x) * px
            return Ex

        def __var_int(self, x, Ex, mu, std):

            px = _normpdf(x, mu, std)
            Vx = (self.ytform.itransform(x) - Ex)**2 * px
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

        def fit(self, X, y, **kwargs):
            y_t = self.le.fit_transform(y)
            return super().fit(X, y_t)

        def get_classes(self):
            tags = [str(c) for c in self.le.classes_]
            return tags

    return EncodedClassifier


#
# Construct compatible classes for the pipeline, these need to be module level
# for pickling...
#

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

    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
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

#
# Helper functions for multiple outputs and missing/masked data
#


def apply_masked(func, data, args=(), kwargs={}):
    # Data is just a matrix (i.e. X for prediction)

    # No masked data
    if np.ma.count_masked(data) == 0:
        return np.ma.array(func(data.data, *args, **kwargs), mask=False)

    # Prediction with missing inputs
    okdata = (data.mask.sum(axis=1)) == 0 if data.ndim == 2 else ~data.mask

    if data.data[okdata].shape[0] == 0:  # if all of this chunk is masked
        # to get dimension of the func return, we create a dummpy res
        res = func(np.ones((1, data.data.shape[1])), *args, **kwargs)
    else:
        res = func(data.data[okdata], *args, **kwargs)

    # For training/fitting that returns nothing
    if not isinstance(res, np.ndarray):
        return res

    # Fill in a padded array the size of the original
    mres = np.empty(len(data)) if res.ndim == 1 \
        else np.empty((len(data), res.shape[1]))

    if data.data[okdata].shape[0] != 0:  # don't change due to dummy res
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

    def unstack(catdata):
        unstck = [d.flatten() if f else d for d, f
                  in zip(np.hsplit(catdata, dims), flat)]
        return unstck

    def unstackfunc(catdata, *nargs, **nkwargs):
        unstckfunc = func(*chain(unstack(catdata), nargs), **nkwargs)
        return unstckfunc

    return apply_masked(unstackfunc, np.ma.hstack(datastack), args, kwargs)


#
# Static module properties
#

# Add all models available to the learning pipeline here!
regressors = {
    'randomforest': RandomForestTransformed,
    'multirandomforest': RandomForestRegressorMulti,
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
    'depthregress': DepthRegressor,
}

classifiers = {
    'logistic': LogisticClassifier,
    'forestclassifier': RandomForestClassifier,
    'svc': SupportVectorClassifier,
    'boostedtrees': GradBoostedTrees
}

modelmaps = {**classifiers, **regressors}

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

    return 1. / (_SQRT2PI * std) * np.exp(-0.5 * ((x - mu) / std)**2)
