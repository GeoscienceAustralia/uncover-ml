import numpy as np
from scipy.integrate import fixed_quad
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, \
    ExpSineSquared
from sklearn.linear_model.stochastic_gradient import SGDRegressor, \
    DEFAULT_EPSILON
from uncoverml.models import RandomForestRegressor, QUADORDER, \
    _normpdf, TagsMixin, SGDApproxGP, basismap
from uncoverml.transforms import target as transforms


class TransformedSGDRegressor(SGDRegressor, TagsMixin):

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
                 verbose=1, epsilon=DEFAULT_EPSILON, random_state=None,
                 learning_rate="invscaling", eta0=0.01, power_t=0.25,
                 warm_start=False, average=False,
                 target_transform='identity'):

        super().__init__(
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

    def fit(self, X, y, *args, **kwargs):
        self.target_transform.fit(y)
        y_t = self.target_transform.transform(y)
        return super().fit(X, y_t)

    def predict(self, X, *args, **kwargs):
        Ey_t = super().predict(X)
        Ey = self.target_transform.itransform(Ey_t)
        return Ey


class TransformedGPRegressor(GaussianProcessRegressor, TagsMixin):

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

        super().__init__(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state
        )

    def fit(self, X, y, *args, **kwargs):
        self.target_transform.fit(y)
        y_t = self.target_transform.transform(y)
        return super().fit(X, y_t)

    def predict(self, X, *args, **kwargs):
        Ey_t = super().predict(X)
        Ey = self.target_transform.itransform(Ey_t)
        return Ey


class TransformedForestRegressor(RandomForestRegressor, TagsMixin):

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
                 warm_start=False
                 ):

        super().__init__(
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

    def fit(self, X, y, *args, **kwargs):
        self.target_transform.fit(y)
        y_t = self.target_transform.transform(y)
        return super().fit(X, y_t)

    def _notransform_predict(self, X, *args, **kwargs):
        Ey = super().predict(X)
        return Ey

    def predict(self, X, *args, **kwargs):
        Ey_t = super().predict(X)
        Ey = self.target_transform.itransform(Ey_t)

        return Ey

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


class TransformedGradientBoost(GradientBoostingRegressor, TagsMixin):

    def __init__(self,
                 target_transform='identity',
                 loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_split=1e-7, init=None,
                 random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto'):

        super().__init__(
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

    def fit(self, X, y, *args, **kwargs):
        self.target_transform.fit(y)
        y_t = self.target_transform.transform(y)
        return super().fit(X, y_t)

    def predict(self, X, *args, **kwargs):
        Ey_t = super().predict(X)
        Ey = self.target_transform.itransform(Ey_t)

        return Ey


transformed_modelmaps = {
    'transformedrandomforest': TransformedForestRegressor,
    'gradientboost': TransformedGradientBoost,
    'transformedgp': TransformedGPRegressor,
    'sgdregressor': TransformedSGDRegressor,
}

# scikit-learn kernels
kernels = {'rbf': RBF,
           'matern': Matern,
           'quadratic': RationalQuadratic,
           }