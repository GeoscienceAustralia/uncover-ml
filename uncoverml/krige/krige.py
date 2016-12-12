import numpy as np
import warnings
import logging
from scipy.stats import norm
from sklearn.base import RegressorMixin, BaseEstimator

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


class KrigePredictProbaMixin():
    """
    Mixin class for providing a ``predict_proba`` method to the
    Krige class.

    This is especially for use with PyKrige Ordinary/UniversalKriging classes.
    """
    def predict_proba(self, x, interval=0.95, *args, **kwargs):
        """
        Predictive mean and variance for a probabilistic regressor.

        Parameters
        ----------
        X: ndarray
            (Ns, d) array query dataset (Ns samples, d dimensions).
        interval: float, optional
            The percentile confidence interval (e.g. 95%) to return.
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

        prediction, variance = \
            self.model.execute('points', x[:, 0], x[:, 1],
                               backend='loop')  # cython backend is not working

        # Determine quantiles
        ql, qu = norm.interval(interval, loc=prediction,
                               scale=np.sqrt(variance))

        return prediction, variance, ql, qu


class Krige(TagsMixin, RegressorMixin, BaseEstimator, KrigePredictProbaMixin):
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
                 verbose=False
                 ):
        if method not in krige_methods.keys():
            raise ConfigException('Kirging method must be '
                                  'one of {}'.format(krige_methods.keys()))
        self.variogram_model = variogram_model
        self.verbose = verbose
        self.nlags = nlags
        self.weight = weight
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
            * args,
            ** kwargs
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
        if not self.model:
            raise Exception('Not trained. Train first')

        if x.shape[1] != 2:
            raise ValueError('krige can use only 2 covariates')

        # cython backend is not working
        return self.model.execute('points', x[:, 0], x[:, 1],
                                  backend='loop',
                                  *args,
                                  ** kwargs)[0]


class MLKrige(Krige):

    def __init__(self,
                 ml_method='transformedrandomforest',
                 ml_params={'n_estimators': 10},
                 method='ordinary',
                 variogram_model='linear',
                 nlags=6,
                 weight=False,
                 verbose=False):
        self.ml_method = ml_method
        self.ml_params = ml_params
        super().__init__(method=method,
                         variogram_model=variogram_model,
                         nlags=nlags,
                         weight=weight,
                         verbose=verbose,
                         )
        self.ml_model = all_ml_models[ml_method](**self.ml_params)
        if not hasattr(self.ml_model, 'predict_proba'):
            raise ConfigException('Only ml_learn algos with a '
                                  'predict_proba method are supported now.')

    def fit(self, x, y, *args, **kwargs):
        self._lat_lon_check(kwargs)
        self.ml_model.fit(x, y)
        ml_pred = self.ml_model.predict(x)
        lat_lon = kwargs['lat_lon']
        super().fit(x=lat_lon, y=y - ml_pred)  # residual=y-ml_pred

    def predict(self, x, *args, **kwargs):
        self._lat_lon_check(kwargs)

        lat_lon = kwargs['lat_lon']
        correction = super(MLKrige, self).predict(lat_lon)
        return self.ml_model.predict(x) + correction

    def _lat_lon_check(self, kwargs):
        if 'lat_lon' not in kwargs:
            raise ValueError('lat_lon must be provided for MLKrige')

    def predict_proba(self, x, interval=0.95, *args, **kwargs):
        """
        must override predict_proba method of Krige class
        Parameters
        ----------
        x
        interval
        args
        kwargs

        Returns
        -------

        """
        self._lat_lon_check(kwargs)
        lat_lon = kwargs['lat_lon']
        correction, corr_var = super(MLKrige, self).predict_proba(
            lat_lon)[:2]
        ml_pred, ml_var = self.ml_model.predict_proba(x, *args, **kwargs)[:2]
        pred = ml_pred + correction
        var = ml_var + corr_var
        # Determine quantiles
        ql, qu = norm.interval(interval, loc=pred,
                               scale=np.sqrt(var))
        return pred, var, ql, qu

krig_dict = {'krige': Krige,
             'mlkrige': MLKrige}
