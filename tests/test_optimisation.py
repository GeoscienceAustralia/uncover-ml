import copy
import numpy as np
import pytest
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from uncoverml.krige import krige_methods, Krige, krig_dict
from uncoverml.optimise.models import kernels
from uncoverml.optimise.models import transformed_modelmaps
from uncoverml.transforms import target as transforms


modelmaps = {**krig_dict, **transformed_modelmaps}

svr = modelmaps.pop('transformedsvr')
krige = modelmaps.pop('krige')
mlkrige = modelmaps.pop('mlkrige')
xgbquantile = modelmaps.pop('xgbquantile')

# TODO: investigate why catboost does not work with target transforms
# catboost = modelmaps.pop('catboost')


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
        param_dict[alg + '__n_estimators'] = [5]
    if hasattr(model(), 'kernel'):
        param_dict[alg + '__kernel'] = [kernel]
    param_dict[alg + '__target_transform'] = [trans]

    estimator = GridSearchCV(pipe,
                             param_dict,
                             n_jobs=1,
                             iid=False,
                             pre_dispatch=2,
                             verbose=True,
                             return_train_score=True,
                             )
    np.random.seed(10)
    estimator.fit(X=1 + np.random.rand(10, 3), y=1. + np.random.rand(10))
    assert estimator.cv_results_['mean_train_score'][0] > -15.0


def test_svr_pipeline(get_transform, get_svr_kernel):
    trans = get_transform()
    pipe = Pipeline(steps=[('svr', svr())])
    param_dict = {'svr__kernel': [get_svr_kernel]}
    param_dict['svr__target_transform'] = [trans]

    estimator = GridSearchCV(pipe,
                             param_dict,
                             n_jobs=1,
                             iid=False,
                             pre_dispatch=2,
                             verbose=True,
                             return_train_score=True,
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
                             iid=False,
                             pre_dispatch=2,
                             verbose=True,
                             return_train_score=True,
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
