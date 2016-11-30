import click
import numpy as np
import warnings
import logging
from scipy.stats import norm

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from uncoverml.logging import warn_with_traceback
from uncoverml.models import TagsMixin

log = logging.getLogger(__name__)
warnings.showwarning = warn_with_traceback


class KrigeMixin():

    def fit(self, x, y, *args, **kwargs):
        """
        Parameters
        ----------
        x: array of Points, (x, y) pairs
        y: ndarray
        """
        if self.method == 'ordinary':
            self.model = OrdinaryKriging(
                x=x[:, 0],
                y=x[:, 1],
                z=y,
                **self.kwargs
             )
        else:
            self.model = UniversalKriging(
                x=x[:, 0],
                y=x[:, 1],
                z=y,
                **self.kwargs
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


class Krige(TagsMixin, KrigeMixin):

    def __init__(self, method, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model = None  # not trained
        self.method = method

krig_dict = {'krige': Krige}
