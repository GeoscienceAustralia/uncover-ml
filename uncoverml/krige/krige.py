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


class Krige(TagsMixin, RegressorMixin, BaseEstimator):

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
        x: array of Points, (x, y) pairs
        y: ndarray
        """

        self.model = krige_methods[self.method](
            x=x[:, 0],
            y=x[:, 1],
            z=y,
            variogram_model=self.variogram_model,
            verbose=self.verbose
         )

    def predict_proba(self, x, interval=0.95, *args, **kwargs):
        prediction, variance = \
            self.model.execute('points', x[:, 0], x[:, 1])

        # Determine quantiles
        ql, qu = norm.interval(interval, loc=prediction,
                               scale=np.sqrt(variance))

        return prediction, variance, ql, qu

    def predict(self, x, interval=0.95):
        return self.predict_proba(x, interval=interval)[0]


krig_dict = {'krige': Krige}
