import numpy as np
import warnings
import logging
from scipy.stats import norm
from sklearn.base import RegressorMixin, BaseEstimator

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from uncoverml.logging import warn_with_traceback
from uncoverml.models import TagsMixin
from uncoverml.config import ConfigException

log = logging.getLogger(__name__)
warnings.showwarning = warn_with_traceback

krige_methods = {'ordinary': OrdinaryKriging,
                 'universal': UniversalKriging}

backend = {'ordinary': 'C',
           'universal': 'loop'}


class KrigePredictProbaMixin():

    def predict_proba(self, x, interval=0.95, *args, **kwargs):
        prediction, variance = \
            self.model.execute('points', x[:, 0], x[:, 1],
                               backend=backend[self.method])

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
                 verbose=False
                 ):
        if method not in krige_methods.keys():
            raise ConfigException('Kirging method must be '
                                  'one of {}'.format(krige_methods.keys()))
        self.variogram_model = variogram_model
        self.verbose = verbose
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
            verbose=self.verbose
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

        return self.model.execute('points', x[:, 0], x[:, 1],
                                  backend=backend[self.method])[0]


krig_dict = {'krige': Krige}
