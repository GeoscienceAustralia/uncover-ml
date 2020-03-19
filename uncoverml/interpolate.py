import logging
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, \
    Rbf, CloughTocher2DInterpolator
from sklearn.base import BaseEstimator, RegressorMixin

_logger = logging.getLogger(__name__)


class SKLearnLinearNDInterpolator(BaseEstimator, RegressorMixin):

    __doc__ = """Scikit-learn wrapper for 
        scipy.interpolate.LinearNDInterpolator class.\n""" + \
        LinearNDInterpolator.__doc__

    def __init__(self, fill_value=0,
                 rescale=False
                 ):

        self.fill_value = fill_value
        self._interpolator = None
        self.rescale = rescale

    def fit(self, X, y):
        self._interpolator = LinearNDInterpolator(
            X, y, fill_value=self.fill_value, rescale=self.rescale)

    def predict(self, X):
        if self._interpolator is None:
            _logger.warning(':mpi:Train first')
            return

        return self._interpolator(X)


class SKLearnNearestNDInterpolator(BaseEstimator, RegressorMixin):

    __doc__ = """Scikit-learn wrapper for 
        scipy.interpolate.NearestNDInterpolator class.\n""" + \
        NearestNDInterpolator.__doc__

    def __init__(self,
                 rescale=False,
                 tree_options=None
                 ):

        self._interpolator = None
        self.rescale = rescale
        self.tree_options = tree_options

    def fit(self, X, y):
        self._interpolator = NearestNDInterpolator(
            X, y, rescale=self.rescale, tree_options=self.tree_options)

    def predict(self, X):
        if self._interpolator is None:
            _logger.warning(':mpi:Train first')
            return
        return self._interpolator(X)


class SKLearnRbf(BaseEstimator, RegressorMixin):

    __doc__ = """Scikit-learn wrapper for scipy.interpolate.Rbf class. \n""" \
              + Rbf.__doc__

    def __init__(self,
                 function='multiquadric',
                 smooth=0,
                 norm='euclidean'
                 ):

        self._interpolator = None
        self.function = function
        self.smooth = smooth
        self.norm = norm

    def fit(self, X, y):
        self._interpolator = Rbf(
            * X.T, y, function=self.function, smooth=self.smooth
        )

    def predict(self, X):
        if self._interpolator is None:
            _logger.warning(':mpi:Train first')
            return

        return self._interpolator(* X.T)


class SKLearnCT(BaseEstimator, RegressorMixin):

    __doc__ = """Scikit-learn wrapper for
        scipy.interpolate.CloughTocher2DInterpolator class.\n""" + \
        CloughTocher2DInterpolator.__doc__

    def __init__(self,
                 fill_value=0,
                 rescale=False,
                 maxiter=1000,
                 tol=1e-4
                 ):

        self._interpolator = None
        self.rescale = rescale
        self.fill_value = fill_value
        self.maxiter = maxiter
        self.tol = tol

    def fit(self, X, y):

        if X.shape[1] != 2:
            raise ValueError('Only 2D interpolation is supported '
                             'via {}'.format(__class__.__name__))

        self._interpolator = CloughTocher2DInterpolator(
            X, y, fill_value=self.fill_value, maxiter=self.maxiter,
            rescale=self.rescale, tol=self.tol
        )

    def predict(self, X):
        if self._interpolator is None:
            _logger.warning(':mpi:Train first')
            return

        return self._interpolator(X)
