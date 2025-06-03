import copy
import numpy as np
import pytest

from unittest.mock import MagicMock, patch
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from uncoverml.krige import krige_methods, Krige, krig_dict
from uncoverml.optimise.models import kernels
from uncoverml.optimise.models import (transformed_modelmaps, 
                                       test_support, 
                                       no_test_support)

from sklearn.linear_model import LinearRegression
from uncoverml.transforms import target as transforms
from uncoverml.optimisation import (score_model, 
                                    r2_score, 
                                    regression_metrics, 
                                    bayesian_optimisation)


modelmaps = {**krig_dict, **test_support}

svr = modelmaps.pop('transformedsvr')
krige = modelmaps.pop('krige')
mlkrige = modelmaps.pop('mlkrige')
xgbquantile = no_test_support.pop('xgbquantile')
xgboostreg = no_test_support.pop('xgbquantileregressor')

# # TODO: investigate why catboost does not work with target transforms
# # catboost = modelmaps.pop('catboost')
#

@pytest.fixture(params=[k for k in modelmaps.keys()])
def get_models(request):
    return request.param, modelmaps[request.param]


@pytest.fixture(params=[k for k in transforms.transforms.keys()])
def get_transform(request):
    return transforms.transforms[request.param]


@pytest.fixture(params=[k for k in kernels.keys()])
def get_kernel(request):
    return kernels[request.param]


@pytest.fixture(params=['linear', 'poly', 'rbf', 'sigmoid'])
def get_svr_kernel(request):
    return request.param


def test_pipeline(get_models, get_transform, get_kernel):

    alg, model = get_models
    trans = get_transform()
    kernel = get_kernel() + WhiteKernel()

    pipe = Pipeline(steps=[(alg, model())])
    param_dict = {}
    if hasattr(model(), 'n_estimators'):
        param_dict[alg + '__n_estimators'] = [2]
    if hasattr(model(), 'kernel'):
        param_dict[alg + '__kernel'] = [kernel]
    param_dict[alg + '__target_transform'] = [trans]

    estimator = GridSearchCV(pipe,
                             param_dict,
                             n_jobs=1,
                             pre_dispatch=2,
                             verbose=True,
                             return_train_score=True,
                             error_score='raise',
                             cv=3,
                             )
    np.random.seed(10)
    estimator.fit(X=1 + np.random.rand(10, 3), y=1. + np.random.rand(10))
    assert estimator.cv_results_['mean_train_score'][0] > -50


def test_xgbquantile_pipeline():

    for alg, model in zip(['xgbquantile'], [xgbquantile]):
        pipe = Pipeline(steps=[(alg, model())])
        param_dict = {}

        estimator = GridSearchCV(pipe,
                                 param_dict,
                                 n_jobs=1,
                                 pre_dispatch=2,
                                 verbose=True,
                                 return_train_score=True,
                                 error_score='raise',
                                 cv=3
                                 )
        np.random.seed(1)
        estimator.fit(X=1 + np.random.rand(10, 5), y=1. + np.random.rand(10))
        assert estimator.cv_results_['mean_train_score'][0] > -10.0


def test_svr_pipeline(get_transform, get_svr_kernel):
    trans = get_transform()
    pipe = Pipeline(steps=[('svr', svr())])
    param_dict = {'svr__kernel': [get_svr_kernel]}
    param_dict['svr__target_transform'] = [trans]

    estimator = GridSearchCV(pipe,
                             param_dict,
                             n_jobs=1,
                             pre_dispatch=2,
                             verbose=True,
                             return_train_score=True,
                             error_score='raise',
                             )
    np.random.seed(1)
    estimator.fit(X=1 + np.random.rand(10, 5), y=1. + np.random.rand(10))
    assert estimator.cv_results_['mean_train_score'][0] > -10.0


@pytest.fixture(params=list(krige_methods.keys()))
def get_krige_method(request):
    return request.param


@pytest.fixture(params=['linear', 'power', 'gaussian', 'spherical',
                        'exponential'])
def get_variogram_model(request):
    return request.param


def test_krige_pipeline(get_krige_method, get_variogram_model):
    pipe = Pipeline(steps=[('krige', Krige(method=get_krige_method))])
    param_dict = {'krige__variogram_model': [get_variogram_model]}

    estimator = GridSearchCV(pipe,
                             param_dict,
                             n_jobs=1,
                             pre_dispatch=2,
                             verbose=True,
                             return_train_score=True,
                             error_score='raise',
                            )
    np.random.seed(1)
    X = np.random.randint(1, 400, size=(20, 2)).astype(float)
    y = 1 + 5*np.random.rand(20)
    estimator.fit(X=X, y=y)
    assert estimator.cv_results_['mean_train_score'][0] > -1.0


def test_gp_std(get_kernel):
    from uncoverml.optimise.models import TransformedGPRegressor
    np.random.seed(10)
    sklearn_gp = TransformedGPRegressor(kernel=get_kernel(length_scale=1))

    sklearn_gp.fit(X=1+np.random.rand(10, 3), y=1 + np.random.rand(10))
    p, v, uq, lq = sklearn_gp.predict_dist(X=1+np.random.rand(5, 3))


@pytest.fixture
def dummy_config(tmp_path):
    return MagicMock(
        algorithm="linear",
        algorithm_args={},
        opt_params_space={
            "fit_intercept": "Categorical([True, False])"
        },
        opt_searchcv_params={
            "n_iter": 2,
            "random_state": 42
        },
        optimised_model_params=str(tmp_path / "params.json"),
        optimisation_output_skopt=str(tmp_path / "optimisation.csv")
    )


@pytest.fixture
def dummy_targets():
    class DummyTargets:
        def __init__(self):
            self.observations = np.random.rand(20)
            self.groups = np.tile(np.arange(5), 4)

    return DummyTargets()


@pytest.fixture
def dummy_data():
    return np.random.rand(20, 5)


@pytest.fixture
def patch_modelmaps():
    with patch("uncoverml.optimisation.modelmaps", {"linear": LinearRegression}):
        yield


@pytest.fixture
def patch_geoio_export():
    with patch("uncoverml.optimisation.geoio.export_model") as mock_export:
        yield mock_export


@pytest.fixture
def patch_bayessearchcv():
    with patch("uncoverml.optimisation.BayesSearchCV") as mock_cls:
        instance = MagicMock()
        instance.best_params_ = {"fit_intercept": True}
        instance.best_score_ = 0.99
        instance.cv_results_ = {
            "rank_test_score": [1],
            "params": [{"fit_intercept": True}],
            "mean_test_score": [0.99]
        }
        mock_cls.return_value = instance
        yield instance


def test_bayesian_optimisation_runs(
    dummy_data, 
    dummy_targets, 
    dummy_config,
    patch_modelmaps, 
    patch_geoio_export, 
    patch_bayessearchcv):
    bayesian_optimisation(dummy_data, dummy_targets, dummy_config)
    patch_bayessearchcv.fit.assert_called_once()
    patch_geoio_export.assert_called_once()
    assert dummy_config.optimised_model is True


def test_score_model_correct():
    model = LinearRegression()
    X = np.random.rand(10, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + 1
    model.fit(X, y)
    scores = score_model(model, X, y)
    assert all(k in scores for k in regression_metrics)
    assert np.isclose(scores["r2_score"], 1.0, atol=1e-6)
