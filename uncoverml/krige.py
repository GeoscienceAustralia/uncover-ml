import numpy as np
import warnings
import logging
from scipy.stats import norm
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.metrics import r2_score

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from uncoverml.mllog import warn_with_traceback
from uncoverml.models import TagsMixin, modelmaps as all_ml_models
from uncoverml.config import ConfigException
from uncoverml.optimise.models import transformed_modelmaps

log = logging.getLogger(__name__)
warnings.showwarning = warn_with_traceback

krige_methods = {'ordinary': OrdinaryKriging,
                 'universal': UniversalKriging}

backend = {'ordinary': 'C',
           'universal': 'loop'}
all_ml_models.update(transformed_modelmaps)


class KrigePredictDistMixin():
    """
    Mixin class for providing a ``predict_dist`` method to the
    Krige class.

    This is especially for use with PyKrige Ordinary/UniversalKriging classes.
    """
    def predict_dist(self, x, interval=0.95, *args, **kwargs):
        """
        Predictive mean and variance for a probabilistic regressor.

        Parameters
        ----------
        x: ndarray
            (Ns, 2) array query dataset (Ns samples, 2 dimensions).
        interval: float, optional
            The percentile confidence interval (e.g. 95%) to return.
        Returns
        -------
        prediction: ndarray
            The expected value of ys for the query inputs, X of shape (Ns,).
        variance: ndarray
            The expected variance of ys (excluding likelihood noise terms) for
            the query inputs, X of shape (Ns,).
        ql: ndarray
            The lower end point of the interval with shape (Ns,)
        qu: ndarray
            The upper end point of the interval with shape (Ns,)
        """
        if not self.model:
            raise Exception('Not trained. Train first')

        if x.shape[1] != 2:
            raise ValueError('krige can use only 2 covariates')

        if isinstance(x, np.ma.masked_array) and np.sum(x.mask):
            x = x.data[x.mask.sum(axis=1) == 0, :]

        # cython backend is not working
        if isinstance(self.model, OrdinaryKriging):
            prediction, variance = \
                self.model.execute('points', x[:, 0], x[:, 1],
                                   n_closest_points=self.n_closest_points,
                                   backend='loop')
        else:
            log.warning('n_closest_point will be ignored for UniversalKriging')
            prediction, variance = \
                self.model.execute('points', x[:, 0], x[:, 1],
                                   backend='loop')

        # Determine quantiles
        ql, qu = norm.interval(interval, loc=prediction,
                               scale=np.sqrt(variance))

        return prediction, variance, ql, qu


class Krige(TagsMixin, RegressorMixin, BaseEstimator, KrigePredictDistMixin):
    """
    A scikitlearn wrapper class for Ordinary and Universal Kriging.
    This works for both Grid/RandomSearchCv for optimising the
    Krige parameters.

    """

    def __init__(self,
                 method='ordinary',
                 variogram_model='linear',
                 nlags=6,
                 weight=False,
                 n_closest_points=10,
                 verbose=False
                 ):
        if method not in krige_methods.keys():
            raise ConfigException('Kirging method must be '
                                  'one of {}'.format(krige_methods.keys()))
        self.variogram_model = variogram_model
        self.verbose = verbose
        self.nlags = nlags
        self.weight = weight
        self.n_closest_points = n_closest_points
        self.model = None  # not trained
        self.method = method

    def fit(self, x, y, *args, **kwargs):
        """
        Parameters
        ----------
        x: ndarray
            array of Points, (x, y) pairs
        y: ndarray
            array of targets
        """
        if x.shape[1] != 2:
            raise ConfigException('krige can use only 2 covariates')

        self.model = krige_methods[self.method](
            x=x[:, 0],
            y=x[:, 1],
            z=y,
            variogram_model=self.variogram_model,
            verbose=self.verbose,
            nlags=self.nlags,
            weight=self.weight,
        )

    def predict(self, x, *args, **kwargs):
        """
        Parameters
        ----------
        x: ndarray

        Returns:
        -------
        Prediction array
        """

        return self.predict_dist(x, *args, **kwargs)[0]


def _check_sklearn_model(model):
    if not (isinstance(model, BaseEstimator) and
            isinstance(model, RegressorMixin)):
        raise RuntimeError('Needs to supply an instance of a scikit-learn '
                           'compatible regression class.')


class MLKrigeBase(TagsMixin):
    """
    This is an implementation of Regression-Kriging as described here:
    https://en.wikipedia.org/wiki/Regression-Kriging
    """

    def __init__(self,
                 ml_method,
                 ml_params={},
                 method='ordinary',
                 variogram_model='linear',
                 n_closest_points=10,
                 nlags=6,
                 weight=False,
                 verbose=False):
        self.n_closest_points = n_closest_points
        self.krige = Krige(method=method,
                           variogram_model=variogram_model,
                           nlags=nlags,
                           weight=weight,
                           n_closest_points=n_closest_points,
                           verbose=verbose,
                           )
        self.ml_model = all_ml_models[ml_method](**ml_params)

        _check_sklearn_model(self.ml_model)

    def fit(self, x, y, lon_lat, *args, **kwargs):
        """
        Fit the ML method and also Krige the residual.

        Parameters
        ----------
        x: ndarray
            (Nt, d) array query dataset (Ns samples, d dimensions)
            for ML regression
        y: ndarray
            array of targets (Nt, )
        lon_lat:
            ndarray of (x, y) points. Needs to be a (Nt, 2) array
            corresponding to the lon/lat, for example.
        """
        self.ml_model.fit(x, y)
        ml_pred = self.ml_model.predict(x)
        # residual=y-ml_pred
        self.krige.fit(x=lon_lat, y=y - ml_pred)

    def predict(self, x, lon_lat, *args, **kwargs):
        """
        Must override predict_dist method of Krige.
        Predictive mean and variance for a probabilistic regressor.

        Parameters
        ----------
        x: ndarray
            (Ns, d) array query dataset (Ns samples, d dimensions)
            for ML regression
        lon_lat:
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.

        Returns
        -------
        pred: ndarray
            The expected value of ys for the query inputs, X of shape (Ns,).
        """
        correction = self.krige_residual(lon_lat)
        ml_pred = self.ml_prediction(x, *args, **kwargs)
        return ml_pred + correction

    def krige_residual(self, lon_lat):
        """
        :param lon_lat:
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.
        :return:
        residual: ndarray
            kriged residual values
        """
        return self.krige.predict(lon_lat)

    def ml_prediction(self, x, *args, **kwargs):
        """
        Parameters
        ----------
        x: ndarray
            regression matrix
        Returns
        -------
            ndarray
            machine learning prediction
        """
        return self.ml_model.predict(x, *args, **kwargs)

    def score(self, x, y, lon_lat, sample_weight=None):
        """
        Overloading default regression score method
        """

        return r2_score(y_pred=self.predict(x, lon_lat),
                        y_true=y,
                        sample_weight=sample_weight)


class MLKrigePredictDistMixin():

    def predict_dist(self, x, interval=0.95, lon_lat=None, *args, **kwargs):
        """
        Predictive mean, variance, lower and upper quantile for a
        probabilistic regressor.

        Parameters
        ----------
        X: ndarray
            (Ns, d) array query dataset (Ns samples, d dimensions)
            for ML regression
        lon_lat:
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.
        interval: float, optional
            The percentile confidence interval (e.g. 95%) to return.
        kwargs must contain a key lon_lat, which needs to be a (Ns, 2) array
        corresponding to the lon/lat
        Returns
        -------
        pred: ndarray
            The expected value of ys for the query inputs, X of shape (Ns,).
        var: ndarray
            The expected variance of ys (excluding likelihood noise terms) for
            the query inputs, X of shape (Ns,).
        ql: ndarray
            The lower end point of the interval with shape (Ns,)
        qu: ndarray
            The upper end point of the interval with shape (Ns,)

        """
        correction, corr_var = self.krige.predict_dist(lon_lat)[:2]
        ml_pred, ml_var = self.ml_model.predict_dist(x, *args, **kwargs)[:2]
        # add prediction and residual
        pred = ml_pred + correction
        # add variances
        var = ml_var + corr_var
        # Determine quantiles
        ql, qu = norm.interval(interval, loc=pred, scale=np.sqrt(var))
        return pred, var, ql, qu


class MLKrigePredictDist(MLKrigeBase, MLKrigePredictDistMixin):
    def __init__(self, *args, **kwargs):
        super(MLKrigePredictDist, self).__init__(*args, **kwargs)


class MLKrige():
    def __new__(cls, ml_method, ml_params={}, *args, **kwargs):
        ml_model = all_ml_models[ml_method](**ml_params)
        if not hasattr(ml_model, 'predict_dist'):
            return MLKrigeBase(ml_method, ml_params, *args, **kwargs)
        else:
            return MLKrigePredictDist(ml_method, ml_params, *args, **kwargs)


krig_dict = {'krige': Krige,
             'mlkrige': MLKrige}
