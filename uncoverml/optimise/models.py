# import sys
# import os
# sys.path.insert(0, os.path.realpath('../../'))

# from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
import logging
import numpy as np
from scipy.integrate import fixed_quad
from scipy.stats import norm, gamma
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.linear_model import (TheilSenRegressor,
                                  RANSACRegressor,
                                  HuberRegressor,
                                  LinearRegression,
                                  ElasticNet)
from sklearn.linear_model.stochastic_gradient import (SGDRegressor,
                                                      DEFAULT_EPSILON)
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from uncoverml.models import RandomForestRegressor, QUADORDER, \
    _normpdf, TagsMixin, SGDApproxGP, PredictProbaMixin, \
    MutualInfoMixin
from revrand.slm import StandardLinearModel
from revrand.basis_functions import LinearBasis
from revrand.btypes import Parameter, Positive
from uncoverml.transforms import target as transforms
# import copy as cp
log = logging.getLogger(__name__)


class TransformMixin():

    def fit(self, X, y, *args, **kwargs):
        self.target_transform.fit(y=y)
        y_t = self.target_transform.transform(y)
        return super().fit(X, y_t)

    def predict(self, X, *args, **kwargs):
        Ey_t = super().predict(X)
        Ey = self.target_transform.itransform(Ey_t)

        return Ey

    def _notransform_predict(self, X, *args, **kwargs):
        Ey = super().predict(X)
        return Ey

    def score(self, X, y, *args, **kwargs):
        """
        This score is used by Scikilearn GridSearchCV/RandomisedSearchCV by
        all models that inherit TransformMixin.
        This is the score as seen by the ML model in the transformed target
        values. The final cross-val score in the original coordinates
        can be obtained from uncoverml.validate.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.

        Returns
        -------
        score : float
            R^2 of self._notransform_predict(X) wrt. y.

        """
        y_t = self.target_transform.transform(y)

        if hasattr(self, 'ml_score') and self.ml_score:
            log.info('Using custom score')
            return r2_score(y_true=y_t,
                            y_pred=self._notransform_predict(
                                X, *args, **kwargs))
        else:
            return super().score(X, y, *args, **kwargs)


class TransformPredictProbaMixin(TransformMixin):

    def __expec_int(self, x, mu, std):
        px = _normpdf(x, mu, std)
        Ex = self.target_transform.itransform(x) * px
        return Ex

    def __var_int(self, x, Ex, mu, std):
        px = _normpdf(x, mu, std)
        Vx = (self.target_transform.itransform(x) - Ex) ** 2 * px
        return Vx

    def predict_proba(self, X, interval=0.95, *args, **kwargs):

        # Expectation and variance in latent space
        Ey_t, Vy_t, ql, qu = super().predict_proba(X, interval)

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


class TransformedLinearReg(TransformPredictProbaMixin, StandardLinearModel,
                           PredictProbaMixin, MutualInfoMixin, TagsMixin):

    def __init__(self,
                 basis=True,
                 var=1.0,
                 regularizer=1.0,
                 tol=1e-8,
                 maxiter=1000,
                 target_transform='identity',
                 ml_score=False,
                 ):
        """
        Parameters
        ----------
        basis : Basis
            A basis object, see the revrand.basis_functions module.
        var : Parameter, optional
            observation variance initial value.
        tol : float, optional
            optimiser function tolerance convergence criterion.
        maxiter : int, optional
            maximum number of iterations for the optimiser.
        target_transform: str, optional
            optional target transform
        ml_score: bool, optional
            whether to use custom score function
        """
        self.regularizer = regularizer
        basis = self.get_basis(basis, regularizer)
        self.basis = basis
        var = self.get_var(var)
        self.var = var
        target_transform = self.get_target_transform(target_transform)
        self.target_transform = target_transform
        self.ml_score = ml_score

        super(TransformedLinearReg, self).__init__(
            basis=basis,
            var=var,
            tol=tol,
            maxiter=maxiter
            )

    def get_basis(self, basis, regulariser):
        # whether to add a bias term
        if isinstance(basis, bool):
            regulariser = self.get_regularizer(regulariser)
            basis = LinearBasis(onescol=basis,
                                regularizer=regulariser)
        return basis

    @staticmethod
    def get_var(var):
        if isinstance(var, float):
            var = gamma(a=var, scale=1)  # Initial target noise
            var = Parameter(var, Positive())
        return var

    @staticmethod
    def get_regularizer(regularizer):
        if isinstance(regularizer, float):
            reg = gamma(a=regularizer, scale=1)  # Initial weight prior
            regularizer = Parameter(reg, Positive())
        return regularizer

    @staticmethod
    def get_target_transform(target_transform):
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        return target_transform


class TransformedSGDRegressor(TransformMixin, SGDRegressor, TagsMixin):

    """
    Linear elastic net regression model using
    Stochastic Gradient Descent (SGD).
    """

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
                 verbose=0, epsilon=DEFAULT_EPSILON, random_state=None,
                 learning_rate="invscaling", eta0=0.01, power_t=0.25,
                 warm_start=False, average=False,
                 target_transform='identity', ml_score=False):

        super(TransformedSGDRegressor, self).__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            n_iter=n_iter,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            warm_start=warm_start,
            average=average,
        )

        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()

        self.target_transform = target_transform
        self.ml_score = ml_score


class TransformedGPRegressor(TransformMixin, GaussianProcessRegressor,
                             TagsMixin):

    def __init__(self,
                 target_transform='identity',
                 kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None,
                 ml_score=False
                 ):

        # uncoverml compatibility if string is passed
        if isinstance(kernel, str):
            kernel = kernels[kernel]()
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()

        self.target_transform = target_transform
        self.ml_score = ml_score

        super(TransformedGPRegressor, self).__init__(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state
        )


class TransformedForestRegressor(TransformPredictProbaMixin,
                                 RandomForestRegressor,
                                 TagsMixin):

    def __init__(self,
                 target_transform='identity',
                 n_estimators=10,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ml_score=False
                 ):

        super(TransformedForestRegressor, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        # training uses str
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()

        # used during optimisation
        self.target_transform = target_transform
        self.ml_score = ml_score


class TransformedGradientBoost(TransformMixin, GradientBoostingRegressor,
                               TagsMixin):

    def __init__(self,
                 target_transform='identity',
                 loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_split=1e-7, init=None,
                 random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto', ml_score=False):

        super(TransformedGradientBoost, self).__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_split=min_impurity_split,
            init=init,
            random_state=random_state,
            max_features=max_features,
            alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            presort=presort,
        )
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform
        self.ml_score = ml_score


class TransformedSVR(TransformMixin, SVR, TagsMixin):

    def __init__(self, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1,
                 target_transform='identity', ml_score=False):
        super(TransformedSVR, self).__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            epsilon=epsilon,
            verbose=verbose,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter)

        # used in training
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()

        self.target_transform = target_transform
        self.ml_score = ml_score


class TransformedSGDApproxGP(TransformMixin, SGDApproxGP, TagsMixin):

    def __init__(self, kernel='rbf', nbases=50, lenscale=1., var=1.,
                 regulariser=1., ard=True, maxiter=3000, batch_size=10,
                 alpha=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8,
                 random_state=None, target_transform='identity',
                 ml_score=False,
                 ):
        super(TransformedSGDApproxGP, self).__init__(
            kernel=kernel,
            nbases=nbases,
            lenscale=lenscale,
            var=var,
            regulariser=regulariser,
            ard=ard,
            maxiter=maxiter,
            batch_size=batch_size,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            random_state=random_state
        )

        # used in training
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform
        self.ml_score = ml_score


class TransformedOLS(TransformMixin, TagsMixin, LinearRegression):

    """
    OLS. Suitable for small learning jobs.
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1, target_transform='identity', ml_score=False):
        # used in training
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform
        self.ml_score = ml_score
        super(TransformedOLS, self).__init__(fit_intercept=fit_intercept,
                                             normalize=normalize,
                                             copy_X=copy_X,
                                             n_jobs=n_jobs)


class TransformedElasticNet(TransformMixin, TagsMixin, ElasticNet):
    """
    Linear regression with combined L1 and L2 priors as regularizer.
    Suitable for small learning jobs.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=0.0001, warm_start=False, positive=False,
                 random_state=None, selection='cyclic',
                 target_transform='identity', ml_score=False):
        # used in training
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform
        self.ml_score = ml_score
        super(TransformedElasticNet, self).__init__(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, max_iter=max_iter,
            copy_X=copy_X, tol=tol, warm_start=warm_start, positive=positive,
            random_state=random_state, selection=selection
            )


class Huber(TransformMixin, TagsMixin, HuberRegressor):
    """
    Robust HuberRegressor
    """

    def __init__(self, epsilon=1.35, max_iter=100, alpha=0.0001,
                 warm_start=False, fit_intercept=True, tol=1e-05,
                 target_transform='identity', ml_score=False):
        # used in training
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform
        self.ml_score = ml_score
        super(Huber, self).__init__(
            epsilon=epsilon, alpha=alpha, fit_intercept=fit_intercept,
            max_iter=max_iter, tol=tol, warm_start=warm_start
            )

# class XGB(LGBMRegressor):
#     def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=10,
#                  max_bin=255, subsample_for_bin=50000, objective=None,
#                  min_split_gain=0, min_child_weight=5, min_child_samples=10,
#                  subsample=1, subsample_freq=1, colsample_bytree=1,
#                  reg_alpha=0, reg_lambda=0, seed=0, nthread=-1, silent=True):
#
#         super(XGB, self).__init__(boosting_type=boosting_type, num_leaves=num_leaves, max_depth=max_depth,
#                                   learning_rate=learning_rate, n_estimators=n_estimators, max_bin=max_bin,
#                                   subsample_for_bin=subsample_for_bin, objective=objective,
#                                   min_split_gain=min_split_gain, min_child_weight=min_child_weight,
#                                   min_child_samples=min_child_samples,
#                                   subsample=subsample, subsample_freq=subsample_freq, colsample_bytree=colsample_bytree,
#                                   reg_alpha=reg_alpha, reg_lambda=reg_lambda, seed=seed, nthread=nthread, silent=silent)
#
#     def fit(self, X, y_t):
#         return super(XGB, self).fit(X, y_t)
#
# class XGBoost(TagsMixin, LGBMRegressor):
#
#     attr_list = ['boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators', 'max_bin',
#                  'subsample_for_bin', 'objective', 'min_split_gain', 'min_child_weight', 'min_child_samples',
#                  'subsample', 'subsample_freq', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'seed', 'nthread', 'silent']
#
#     def __init__(self, target_transform='identity', ml_score=False, boosting_type="gbdt", num_leaves=31, max_depth=-1,
#         learning_rate=0.1, n_estimators=10, max_bin=255,
#         subsample_for_bin=50000, objective=None,
#         min_split_gain=0, min_child_weight=5, min_child_samples=10,
#         subsample=1, subsample_freq=1, colsample_bytree=1,
#         reg_alpha=0, reg_lambda=0, seed=0, nthread=-1, silent=True):
#
#         if isinstance(target_transform, str):
#             target_transform = transforms.transforms[target_transform]()
#         self.target_transform = target_transform
#         self.ml_score = ml_score
#
#         super(XGBoost,self).__init__(boosting_type=boosting_type, num_leaves=num_leaves, max_depth=max_depth,
#             learning_rate=learning_rate, n_estimators=n_estimators, max_bin=max_bin,
#             subsample_for_bin=subsample_for_bin, objective=objective,
#             min_split_gain=min_split_gain, min_child_weight=min_child_weight, min_child_samples=min_child_samples,
#             subsample=subsample, subsample_freq=subsample_freq, colsample_bytree=colsample_bytree,
#             reg_alpha=reg_alpha, reg_lambda=reg_lambda, seed=seed, nthread=nthread, silent=silent)
#
#     def fit(self, X, y, *args, **kwargs):
#         self.target_transform.fit(y=y)
#         y_t = self.target_transform.transform(y)
#         result = dict((attr,val) for attr, val in self.__dict__.items() if attr in self.attr_list)
#         xgb = XGB(**result)
#         # self.__delattr__('ml_score')
#         # self.__delattr__('target_transform')
#
#         # a = self.__dict__
#         # for attr, val in self.__dict__.iteritems():
#         #     print(attr, val)
#         # for attr, val in self.__dict__.__delitem__()
#         # self_cpy = cp..deepcopy(self)
#         print(xgb)
#         a=xgb.fit(X, y_t)
#         print(xgb)
#         print(a)
#
#         # delattr(self, 'target_transform')
#         # delattr(self, 'ml_score')
#         #


class XGBoost(TransformMixin, TagsMixin, XGBRegressor):

    def __init__(self, target_transform='identity', ml_score=False, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None):

        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
            self.target_transform = target_transform
            self.ml_score = ml_score

            # super(XGBoost, self).__init__(max_depth=3, learning_rate=0.1, n_estimators=100,
            #      silent=True, objective="reg:linear", booster='gbtree',
            #      n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
            #      subsample=1, colsample_bytree=1, colsample_bylevel=1,
            #      reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
            #      base_score=0.5, random_state=0, seed=None, missing=None)

            super(XGBoost, self).__init__(max_depth, learning_rate, n_estimators,
                                          silent, objective, booster,
                                          n_jobs, nthread, gamma, min_child_weight, max_delta_step,
                                          subsample, colsample_bytree, colsample_bylevel,
                                          reg_alpha, reg_lambda, scale_pos_weight,
                                          base_score, random_state, seed, missing)





transformed_modelmaps = {
    'transformedrandomforest': TransformedForestRegressor,
    'gradientboost': TransformedGradientBoost,
    'transformedgp': TransformedGPRegressor,
    'sgdregressor': TransformedSGDRegressor,
    'transformedsvr': TransformedSVR,
    'transformedbayesreg': TransformedLinearReg,
    'ols': TransformedOLS,
    'elasticnet': TransformedElasticNet,
    'huber': Huber,
    'xgboost': XGBoost,
}

# scikit-learn kernels
kernels = {'rbf': RBF,
           'matern': Matern,
           'quadratic': RationalQuadratic,
           }
