import logging
from functools import partial
import numpy as np
from scipy.integrate import fixed_quad
from scipy.stats import norm, gamma
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.linear_model import (HuberRegressor,
                                  LinearRegression,
                                  ElasticNet, SGDRegressor)
from sklearn.linear_model._stochastic_gradient import DEFAULT_EPSILON
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from uncoverml.models import RandomForestRegressor, QUADORDER, \
    _normpdf, TagsMixin, SGDApproxGP, SVRTransformed
from uncoverml.transforms import target as transforms

log = logging.getLogger(__name__)


class TransformPredictMixin:

    def predict(self, X, *args, **kwargs):
        Ey_t = self._notransform_predict(X, *args, **kwargs)
        return self.target_transform.itransform(Ey_t)

    def _notransform_predict(self, X, *args, **kwargs):
        Ey_t = super().predict(X)
        return Ey_t


class TransformMixin(TransformPredictMixin):

    def fit(self, X, y, *args, **kwargs):
        self.target_transform.fit(y=y)
        y_t = self.target_transform.transform(y)
        return super().fit(X, y_t, sample_weight=kwargs['sample_weight'])


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


class TransformedGPRegressor(TransformPredictDistMixin, GaussianProcessRegressor,
                             TagsMixin):

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

        return Ey, std_t**2, ql, qu

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
                 min_impurity_decrease=1e-7,
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
            min_impurity_decrease=min_impurity_decrease,
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


class TransformedSVR(TransformMixin, SVR, TagsMixin):

    def __init__(self, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
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


class XGBoost(XGBRegressor, TagsMixin):

    def __init__(self,
                 target_transform='identity',
                 max_depth=None,
                 learning_rate=None,
                 n_estimators=100,
                 verbosity=None,
                 objective=None,
                 booster=None,
                 tree_method=None,
                 n_jobs=None,
                 gamma=None,
                 min_child_weight=None,
                 max_delta_step=None,
                 subsample=None,
                 colsample_bytree=None,
                 colsample_bylevel=None,
                 colsample_bynode=None,
                 reg_alpha=None,
                 reg_lambda=None,
                 scale_pos_weight=None,
                 base_score=None,
                 random_state=None,
                 missing=np.nan,
                 num_parallel_tree=None,
                 monotone_constraints=None,
                 interaction_constraints=None,
                 importance_type="gain",
                 gpu_id=None,
                 validate_parameters=None,
                 ):

        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform

        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            max_depth=max_depth,
            learning_rate=learning_rate,
            verbosity=verbosity,
            booster=booster,
            tree_method=tree_method,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            missing=missing,
            num_parallel_tree=num_parallel_tree,
            random_state=random_state,
            n_jobs=n_jobs,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            importance_type=importance_type,
            gpu_id=gpu_id,
            validate_parameters=validate_parameters
        )

    def fit(self, X, y, *args, **kwargs):
        self.target_transform.fit(y=y)
        y_t = self.target_transform.transform(y)
        sample_weight = kwargs['sample_weight']
        return super().fit(X, y_t, sample_weight=sample_weight)

    def predict(self, X, *args, **kwargs):
        Ey_t = self._notransform_predict(X, *args, **kwargs)
        return self.target_transform.itransform(Ey_t)

    def _notransform_predict(self, X, *args, **kwargs):
        Ey_t = super().predict(X, *args)
        return Ey_t


class XGBQuantileRegressor(XGBRegressor, TagsMixin):

    def __init__(self,
                 target_transform='identity',
                 alpha=0.95, delta=1.0, thresh=1.0, variance=1.0,
                 **kwargs
                 ):
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform
        self.alpha = alpha
        self.delta = delta
        self.thresh = thresh
        self.variance = variance
        if 'objective' in kwargs:
            kwargs.pop('objective')
        super().__init__(**kwargs)

    def fit(self, X, y, * args, **kwargs):
        quantile_loss_obj = partial(self.quantile_loss,
                                    alpha=self.alpha,
                                    delta=self.delta,
                                    threshold=self.thresh,
                                    var=self.variance)

        super().set_params(objective=quantile_loss_obj)
        self.target_transform.fit(y=y)
        y_t = self.target_transform.transform(y)
        super().fit(X, y_t, sample_weight=kwargs['sample_weight'])
        return self

    def predict(self, X, *args, **kwargs):
        Ey_t = self._notransform_predict(X, *args, **kwargs)
        return self.target_transform.itransform(Ey_t)

    def _notransform_predict(self, X, *args, **kwargs):
        Ey_t = super().predict(X, *args)
        return Ey_t

    @staticmethod
    def quantile_loss(y_true, y_pred, alpha, delta, threshold, var):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - \
               ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - \
               alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta

        grad = (np.abs(x) < threshold) * grad - (np.abs(x) >= threshold) * (
                2 * np.random.randint(2, size=len(y_true)) - 1.0) * var
        hess = (np.abs(x) < threshold) * hess + (np.abs(x) >= threshold)
        return grad, hess

    # def score(self, X, y, **kwargs):
    #     y_pred = super().predict(X)
    #     score = self.quantile_score(y, y_pred, self.alpha)
    #     score = 1. / score
    #     return score
    #
    # @staticmethod
    # def quantile_score(y_true, y_pred, alpha):
    #     score = XGBQuantileRegressor.quantile_cost(x=y_true - y_pred, alpha=alpha)
    #     score = np.sum(score)
    #     return score
    #
    # @staticmethod
    # def quantile_cost(x, alpha):
    #     return (alpha - 1.0) * x * (x < 0) + alpha * x * (x >= 0)
    #
    # @staticmethod
    # def get_split_gain(gradient, hessian, l=1):
    #     split_gain = list()
    #     for i in range(gradient.shape[0]):
    #         split_gain.append(np.sum(gradient[:i]) / (np.sum(hessian[:i]) + l) + np.sum(gradient[i:]) / (
    #                 np.sum(hessian[i:]) + l) - np.sum(gradient) / (np.sum(hessian) + l))
    #
    #     return np.array(split_gain)


class QuantileXGB(TagsMixin, BaseEstimator, RegressorMixin):
    def __init__(
            self,
            mean_model_params={},
            upper_quantile_params={'alpha': 0.95, 'delta': 1.0, 'thresh': 1.0, 'variance': 1.0},
            lower_quantile_params={'alpha': 0.05, 'delta': 1.0, 'thresh': 1.0, 'variance': 1.0}
    ):
        self.mean_model_params = mean_model_params
        self.upper_quantile_params = upper_quantile_params
        self.lower_quantile_params = lower_quantile_params
        self.gb = XGBRegressor(**mean_model_params)
        mean_model_params.pop('alpha', None)
        upper_quantile_params_combined = {**mean_model_params}
        upper_quantile_params_combined.update(upper_quantile_params)
        lower_quantile_params_combiled = {**mean_model_params}
        lower_quantile_params_combiled.update(lower_quantile_params)
        self.gb_quantile_upper = XGBQuantileRegressor(**upper_quantile_params_combined)
        self.gb_quantile_lower = XGBQuantileRegressor(**lower_quantile_params_combiled)
        self.upper_alpha = upper_quantile_params['alpha']
        self.lower_alpha = lower_quantile_params['alpha']

    @staticmethod
    def collect_prediction(regressor, X_test):
        y_pred = regressor.predict(X_test)
        return y_pred

    def fit(self, X, y, **kwargs):
        log.info('Fitting xgb base model')
        self.gb.fit(X, y, sample_weight=kwargs['sample_weight'])
        log.info('Fitting xgb upper quantile model')
        self.gb_quantile_upper.fit(X, y, sample_weight=kwargs['sample_weight'])
        log.info('Fitting xgb lower quantile model')
        self.gb_quantile_lower.fit(X, y, sample_weight=kwargs['sample_weight'])

    def predict(self, X, *args, **kwargs):
        return self.predict_dist(X, *args, **kwargs)[0]

    def predict_dist(self, X, interval=0.95, *args, **kwargs):
        Ey = self.gb.predict(X)

        ql_ = self.collect_prediction(self.gb_quantile_lower, X)  # upper quantile model prediction
        qu_ = self.collect_prediction(self.gb_quantile_upper, X)  # lower quantile model prediction
        # divide qu - ql by the normal distribution Z value diff between the quantiles, square for variance
        Vy = ((qu_ - ql_) / (norm.ppf(self.upper_alpha) - norm.ppf(self.lower_alpha))) ** 2

        # to make gbm quantile model consistent with other quantile based models
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


class GBMReg(GradientBoostingRegressor, TagsMixin):
    def __init__(self, target_transform='identity', loss='quantile', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.5, verbose=0, max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):

        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()
        self.target_transform = target_transform

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, alpha=alpha, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha
        )

    def predict(self, X, *args, **kwargs):
        Ey_t = self._notransform_predict(X, *args, **kwargs)
        return self.target_transform.itransform(Ey_t)

    def _notransform_predict(self, X, *args, **kwargs):
        Ey_t = super().predict(X)
        return Ey_t

    def fit(self, X, y, *args, **kwargs):
        self.target_transform.fit(y=y)
        y_t = self.target_transform.transform(y)
        return super().fit(X, y_t, sample_weight=kwargs['sample_weight'])


class QuantileGradientBoosting(BaseEstimator, RegressorMixin, TagsMixin):
    def __init__(self, target_transform='identity', loss='quantile',
                 alpha=0.5, upper_alpha=0.95, lower_alpha=0.05,
                 learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, verbose=0, max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0
                 ):
        log.warn(f"Supplied loss: {loss} and alpha {alpha} are not going to be used")
        if isinstance(target_transform, str):
            target_transform = transforms.transforms[target_transform]()

        self.target_transform = target_transform
        # loss = 'quantile'  # use quantile loss for median
        # alpha = 0.5  # median
        self.median_quantile_params ={'loss': 'quantile', 'alpha': 0.5}
        self.upper_quantile_params = {'loss': 'quantile', 'alpha': upper_alpha}
        self.lower_quantile_params = {'loss': 'quantile', 'alpha': lower_alpha}
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha

        self.gb = GradientBoostingRegressor(
            learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha,
            **self.median_quantile_params
        )
        self.gb_quantile_upper = GradientBoostingRegressor(
            learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha,
            **self.upper_quantile_params
        )
        self.gb_quantile_lower = GradientBoostingRegressor(
            learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha,
            **self.lower_quantile_params
        )
        self.upper_alpha = upper_alpha
        self.lower_alpha = lower_alpha

    @staticmethod
    def collect_prediction(regressor, X_test):
        y_pred = regressor.predict(X_test)
        return y_pred

    def fit(self, X, y, *args, **kwargs):
        log.info('Fitting gb base model')
        self.gb.fit(X, y, sample_weight=kwargs['sample_weight'])
        log.info('Fitting gb upper quantile model')
        self.gb_quantile_upper.fit(X, y, sample_weight=kwargs['sample_weight'])
        log.info('Fitting gb lower quantile model')
        self.gb_quantile_lower.fit(X, y, sample_weight=kwargs['sample_weight'])

    def predict(self, X, *args, **kwargs):
        return self.predict_dist(X, *args, **kwargs)[0]

    def predict_dist(self, X, interval=0.95, *args, ** kwargs):
        Ey = self.gb.predict(X)

        ql_ = self.collect_prediction(self.gb_quantile_lower, X)
        qu_ = self.collect_prediction(self.gb_quantile_upper, X)
        # divide qu - ql by the normal distribution Z value diff between the quantiles, square for variance
        Vy = ((qu_ - ql_) / (norm.ppf(self.upper_alpha) - norm.ppf(self.lower_alpha))) ** 2

        # to make gbm quantile model consistent with other quantile based models
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


class CatBoostWrapper(CatBoostRegressor, TagsMixin):

    def __init__(self,  **kwargs):
        if 'loss_function' in kwargs:
            kwargs.pop('loss_function')
            log.warn("For uncertainty estimation we are going to use 'RMSEWithUncertainty' loss!\n"
                     "Supplied loss function was not used!!!")
        super(CatBoostWrapper, self).__init__(**kwargs, loss_function='RMSEWithUncertainty')

    def fit(self, X, y, **kwargs):
        super().fit(X, y, sample_weight=kwargs['sample_weight'])

    def predict(self, X, *args, **kwargs):
        return self.predict_dist(X, *args, **kwargs)[0]

    def predict_dist(self, X, interval=0.95, **kwargs):
        pred = super().predict(X)
        Ey = pred[:, 0]
        Vy = pred[:, 1]
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))
        return Ey, Vy, ql, qu


no_test_support = {
    'xgboost': XGBoost,
    'xgbquantileregressor': XGBQuantileRegressor,
    'xgbquantile': QuantileXGB,
    'quantilegb': QuantileGradientBoosting,
    'gradientboost': GBMReg,
    'catboost': CatBoostWrapper,
    'svr': SVRTransformed,
}

test_support = {
    'transformedrandomforest': TransformedForestRegressor,
    'transformedgp': TransformedGPRegressor,
    'sgdregressor': TransformedSGDRegressor,
    'transformedsvr': TransformedSVR,
    'ols': TransformedOLS,
    'elasticnet': TransformedElasticNet,
    'huber': Huber,
}

transformed_modelmaps = {** test_support, ** no_test_support}

# scikit-learn kernels
kernels = {'rbf': RBF,
           'matern': Matern,
           'quadratic': RationalQuadratic,
           }
