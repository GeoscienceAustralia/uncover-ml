"""
12-05-2020 15:50:35 AEST - brenainn.moushall@ga.gov.au

.. note:: 

    Only some models are compatible with optimisation. This is because
    models must be structued in a way compatible with scikit-learn's
    GridSearchCV. This involves:

    - having all arguments explicitly listed in the ``__init__`` signature (no varargs)
    - having the expected functions (``fit``, ``predict``, etc.)
    - implemeting the ``get_params`` and ``set_params`` functions
      defined by `Base Estimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`_


# TODO: refactor all models to have an interface compatible with GCV
# and consolidate to a single module.
"""
import logging
import inspect
from functools import partial
import numpy as np
from scipy.integrate import fixed_quad
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.linear_model import (HuberRegressor,
                                  LinearRegression,
                                  ElasticNet,
                                  SGDRegressor)
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from uncoverml.models import RandomForestRegressor, QUADORDER, \
    _normpdf, TagsMixin, SGDApproxGP
from uncoverml.transforms import target as transforms

log = logging.getLogger(__name__)

# from sklearn.linear_model._stochastic_gradient
DEFAULT_EPSILON = 0.1


class TransformMixin:

    def fit(self, X, y, *args, **kwargs):
        self.target_transform.fit(y=y)
        y_t = self.target_transform.transform(y)
        # Hack to check if we can apply sample weights
        if 'sample_weight' in inspect.signature(super().fit).parameters.keys() \
                and 'sample_weight' in kwargs.keys():
            return super().fit(X, y_t, sample_weight=kwargs['sample_weight'])
        else:
            return super().fit(X, y_t)

    def predict(self, X, *args, **kwargs):

        if 'return_std' in kwargs:
            return_std = kwargs.pop('return_std')
            if return_std:
                Ey_t, std_t = super().predict(X, return_std=return_std)

                return self.target_transform.itransform(Ey_t), \
                       self.target_transform.itransform(std_t)

        Ey_t = self._notransform_predict(X, *args, **kwargs)
        return self.target_transform.itransform(Ey_t)

    def _notransform_predict(self, X, *args, **kwargs):
        Ey_t = super().predict(X)
        return Ey_t


class TransformPredictDistMixin(TransformMixin):

    def __expec_int(self, x, mu, std):
        px = _normpdf(x, mu, std)
        Ex = self.target_transform.itransform(x) * px
        return Ex

    def __var_int(self, x, Ex, mu, std):
        px = _normpdf(x, mu, std)
        Vx = (self.target_transform.itransform(x) - Ex) ** 2 * px
        return Vx

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


class TransformedSGDRegressor(TransformPredictDistMixin, SGDRegressor, TagsMixin):
    """
    Linear elastic net regression model using
    Stochastic Gradient Descent (SGD).
    """

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, max_iter=5, shuffle=True,
                 verbose=0, epsilon=DEFAULT_EPSILON, random_state=None,
                 learning_rate="invscaling", eta0=0.01, power_t=0.25,
                 warm_start=False, average=False,
                 target_transform='identity'):
        super(TransformedSGDRegressor, self).__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
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


class TransformedGPRegressor(TransformPredictDistMixin, GaussianProcessRegressor, TagsMixin):

    def __init__(self,
                 target_transform='identity',
                 kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None
                 ):

        # uncoverml compatibility if string is passed
        if isinstance(kernel, str):
            kernel = kernels[kernel]()
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()

        self.target_transform = target_transform

        super(TransformedGPRegressor, self).__init__(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state
        )

    def predict_dist(self, X, interval=0.95, *args, **kwargs):
        Ey, std_t = super().predict(X, return_std=True)
        ql, qu = norm.interval(interval, loc=Ey, scale=std_t)

        return Ey, std_t ** 2, ql, qu

    def predict(self, X, *args, **kwargs):
        return self.predict_dist(X)[0]


class TransformedForestRegressor(TransformPredictDistMixin,
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
                 warm_start=False, presort='auto'):
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


class TransformedSVR(TransformMixin, SVR, TagsMixin):

    def __init__(self, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
                 tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1,
                 target_transform='identity'):
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


class TransformedSGDApproxGP(TransformMixin, SGDApproxGP, TagsMixin):

    def __init__(self, kernel='rbf', nbases=50, lenscale=1., var=1.,
                 regulariser=1., ard=True, maxiter=3000, batch_size=10,
                 alpha=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8,
                 random_state=None, target_transform='identity'
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


class TransformedOLS(TransformMixin, TagsMixin, LinearRegression):
    """
    OLS. Suitable for small learning jobs.
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1, target_transform='identity'):
        # used in training
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform
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
                 target_transform='identity'):
        # used in training
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform

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
                 target_transform='identity'):
        # used in training
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform
        super(Huber, self).__init__(
            epsilon=epsilon, alpha=alpha, fit_intercept=fit_intercept,
            max_iter=max_iter, tol=tol, warm_start=warm_start
        )


class XGBoost(TransformMixin, TagsMixin, XGBRegressor):

    def __init__(self, target_transform='identity',
                 max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear", booster='gbtree',
                 nthread=1, gamma=0, min_child_weight=1,
                 max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, n_jobs=-1,
                 base_score=0.5, random_state=1, missing=None, eval_metric='rmse', tree_method='auto', seed=1):
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform

        super(XGBoost, self).__init__(max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      n_estimators=n_estimators,
                                      silent=silent,
                                      booster=booster,
                                      objective=objective,
                                      nthread=nthread,
                                      gamma=gamma,
                                      min_child_weight=min_child_weight,
                                      max_delta_step=max_delta_step,
                                      subsample=subsample,
                                      colsample_bytree=colsample_bytree,
                                      colsample_bylevel=colsample_bylevel,
                                      reg_alpha=reg_alpha,
                                      reg_lambda=reg_lambda,
                                      scale_pos_weight=scale_pos_weight,
                                      base_score=base_score,
                                      random_state=random_state,
                                      missing=missing,
                                      n_jobs=n_jobs,
                                      eval_metric=eval_metric,
                                      tree_method=tree_method,
                                      seed=seed)


class XGBQuantile(XGBoost):
    def __init__(self, target_transform='identity',
                 quant_alpha=0.95, quant_delta=1.0, quant_thres=1.0, quant_var=1.0, base_score=0.5,
                 booster='gbtree', colsample_bylevel=1,
                 colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1,
                 missing=None, n_estimators=100,
                 n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1,
                 scale_pos_weight=1, seed=None, silent=True, subsample=1, eval_metric='rmse', tree_method='auto'):
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var

        super(XGBQuantile, self).__init__(
            target_transform=target_transform,
            base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
            colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate,
            max_delta_step=max_delta_step,
            max_depth=max_depth, min_child_weight=min_child_weight, missing=missing,
            n_estimators=n_estimators,
            n_jobs=n_jobs, nthread=nthread, objective=objective, random_state=random_state,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, seed=seed,
            silent=silent, subsample=subsample, eval_metric=eval_metric, tree_method=tree_method
        )

    def fit(self, X, y, **kwargs):
        super().set_params(objective=partial(XGBQuantile.quantile_loss, alpha=self.quant_alpha, delta=self.quant_delta,
                                             threshold=self.quant_thres, var=self.quant_var))
        super().fit(X, y)
        return self

    def predict(self, X, **kwargs):
        return super().predict(X)

    def score(self, X, y, **kwargs):
        y_pred = super().predict(X)
        score = XGBQuantile.quantile_score(y, y_pred, self.quant_alpha)
        score = 1. / score
        return score

    @staticmethod
    def quantile_loss(y_true, y_pred, alpha, delta, threshold, var):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta

        grad = (np.abs(x) < threshold) * grad - (np.abs(x) >= threshold) * (
                2 * np.random.randint(2, size=len(y_true)) - 1.0) * var
        hess = (np.abs(x) < threshold) * hess + (np.abs(x) >= threshold)
        return grad, hess

    @staticmethod
    def original_quantile_loss(y_true, y_pred, alpha, delta):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta
        return grad, hess

    @staticmethod
    def quantile_score(y_true, y_pred, alpha):
        score = XGBQuantile.quantile_cost(x=y_true - y_pred, alpha=alpha)
        score = np.sum(score)
        return score

    @staticmethod
    def quantile_cost(x, alpha):
        return (alpha - 1.0) * x * (x < 0) + alpha * x * (x >= 0)

    @staticmethod
    def get_split_gain(gradient, hessian, l=1):
        split_gain = list()
        for i in range(gradient.shape[0]):
            split_gain.append(np.sum(gradient[:i]) / (np.sum(hessian[:i]) + l) + np.sum(gradient[i:]) / (
                    np.sum(hessian[i:]) + l) - np.sum(gradient) / (np.sum(hessian) + l))

        return np.array(split_gain)


class XGBQuantileWrapper(TagsMixin):
    def __init__(
            self,
            quant_alpha=0.95,
            quant_delta_upper=1.0, quant_thres_upper=1.0, quant_var_upper=1.0,
            quant_delta_lower=1.0, quant_thres_lower=1.0, quant_var_lower=1.0,
            **kwargs):
        self.quant_alpha = quant_alpha
        self.xgboost = XGBoost(** kwargs)
        self.xgboost_quantile_upper = XGBQuantile(
            quant_alpha=quant_alpha, quant_delta=quant_delta_upper,
            quant_thres=quant_thres_upper, quant_var=quant_var_upper,
            ** kwargs
        )
        self.xgboost_quantile_lower = XGBQuantile(
            quant_alpha=1-quant_alpha, quant_delta=quant_delta_lower,
            quant_thres=quant_thres_lower, quant_var=quant_var_lower,
            ** kwargs
        )

    @staticmethod
    def collect_prediction(regressor, X_test):
        y_pred = regressor.predict(X_test)
        return y_pred

    def fit(self, X, y, **kwargs):
        self.xgboost.fit(X, y, **kwargs)
        self.xgboost_quantile_upper.fit(X, y, **kwargs)
        self.xgboost_quantile_lower.fit(X, y, **kwargs)

    def predict(self, X):
        return self.predict_dist(X)[0]

    def predict_dist(self, X, *args, **kwargs):
        Ey = self.xgboost.predict(X)

        ql = self.collect_prediction(self.xgboost_quantile_lower, X)
        qu = self.collect_prediction(self.xgboost_quantile_upper, X)
        # divide qu - ql by the normal distribution Z value diff between the quantiles, square for variance
        Vy = ((qu - ql)/(norm.ppf(self.quant_alpha) - norm.ppf(1-self.quant_alpha))) ** 2

        return Ey, Vy, ql, qu


transformed_modelmaps = {
    'transformedrandomforest': TransformedForestRegressor,
    'gradientboost': TransformedGradientBoost,
    'transformedgp': TransformedGPRegressor,
    'sgdregressor': TransformedSGDRegressor,
    'transformedsvr': TransformedSVR,
    'ols': TransformedOLS,
    'elasticnet': TransformedElasticNet,
    'huber': Huber,
    'xgboost': XGBoost,
    'xgbquantile': XGBQuantileWrapper
}

# scikit-learn kernels
kernels = {'rbf': RBF,
           'matern': Matern,
           'quadratic': RationalQuadratic,
           }
