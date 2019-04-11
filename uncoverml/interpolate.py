import logging
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, Rbf
from sklearn.base import BaseEstimator, RegressorMixin

log = logging.getLogger(__name__)


class SKLearnLinearNDInterpolator(BaseEstimator, RegressorMixin):

    __doc__ = """Scikit-learn wrapper for Rbf class. \n""" + \
              LinearNDInterpolator.__doc__

    def __init__(self, fill_value=0,
                 rescale=False
                 ):

        log.info("Using Linear Interpolation")

        self.fill_value = fill_value
        self.interpolator = None
        self.rescale = rescale

    def fit(self, X, y):
        log.info("Fitting linear interpolation")
        self.interpolator = LinearNDInterpolator(
            X, y, fill_value=self.fill_value, rescale=self.rescale)

    def predict(self, X):
        log.info("predicting using linear interplocation")
        if self.interpolator is None:
            print('Train first')
            return

        return self.interpolator(X)


class SKLearnNearestNDInterpolator(BaseEstimator, RegressorMixin):

    __doc__ = """Scikit-learn wrapper for Rbf class. \n""" + \
              NearestNDInterpolator.__doc__

    def __init__(self,
                 rescale=False,
                 tree_options=None
                 ):

        self.interpolator = None
        self.rescale = rescale
        self.tree_options = tree_options

    def fit(self, X, y):
        self.interpolator = NearestNDInterpolator(
            X, y, rescale=self.rescale, tree_options=self.tree_options)

    def predict(self, X):
        if self.interpolator is None:
            print('Train first')
            return

        return self.interpolator(X)


class SKLearnRbf(BaseEstimator, RegressorMixin):

    __doc__ = """Scikit-learn wrapper for Rbf class. \n""" + Rbf.__doc__

    def __init__(self,
                 function='multiquadric',
                 smooth=0,
                 norm='euclidean'
                 ):

        self.interpolator = None
        self.function = function
        self.smooth = smooth
        self.norm = norm

    def fit(self, X, y):
        self.interpolator = Rbf(
            * X.T, y, function=self.function, smooth=self.smooth
        )

    def predict(self, X):
        if self.interpolator is None:
            print('Train first')
            return

        return self.interpolator(* X.T)


# interpolators = {
#     'linear': SKLearnLinearNDInterpolator,
#     'nn': SKLearnNearestNDInterpolator,
#     'rbf': SKLearnRbf
# }
#
#
# if __name__ == '__main__':
#     import numpy as np
#     x = np.array([-4386795.73911443, -1239996.25110694, -3974316.43669208,
#                   1560260.49911342,  4977361.53694849, -1996458.01768192,
#                   5888021.46423068,  2969439.36068243,   562498.56468588,
#                   4940040.00457585])
#
#     y = np.array([-572081.11495993, -5663387.07621326,  3841976.34982795,
#                   3761230.61316845,  -942281.80271223,  5414546.28275767,
#                   1320445.40098735, -4234503.89305636,  4621185.12249923,
#                   1172328.8107458])
#
#     z = np.array([ 4579159.6898615 ,  2649940.2481702 ,  3171358.81564312,
#                    4892740.54647532,  3862475.79651847,  2707177.605241  ,
#                    2059175.83411223,  3720138.47529587,  4345385.04025412,
#                    3847493.83999694])
#
#     X = np.vstack([x, y]).T
#     model = interpolators['rbf']()
#
#     model.fit(X, z)
#
#     Xp = np.linspace(min(x), max(x), 5)
#     Yp = np.linspace(min(y), max(y), 5)
#     Xs, Ys = np.meshgrid(Xp, Yp)
#
#     #
#     Xc = np.vstack([Xs.flatten(), Ys.flatten()])
#
#     print(Xp.shape, Xs.shape, Xc.shape)
#     di = model.predict(Xc)
#     print(di.shape)
