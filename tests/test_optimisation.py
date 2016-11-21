import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import WhiteKernel

from uncoverml.optimise.pipeline import transformed_modelmaps
from uncoverml.transforms import target as transforms
from uncoverml.optimise.models import kernels

svr = transformed_modelmaps.pop('transformedsvr')

@pytest.fixture(params=[k for k in transformed_modelmaps.keys()])
def get_models(request):
    return request.param, transformed_modelmaps[request.param]


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
                             )
    estimator.fit(X=1 + np.random.rand(10, 5), y=1. + np.random.rand(10))
    assert estimator.cv_results_['mean_train_score'][0] > -100.0


def test_svr_pipeline(get_transform, get_svr_kernel):

    trans = get_transform()
    pipe = Pipeline(steps=[('svr', svr())])
    param_dict = {'svr' + '__kernel': [get_svr_kernel]}
    param_dict['svr' + '__target_transform'] = [trans]

    estimator = GridSearchCV(pipe,
                             param_dict,
                             n_jobs=1,
                             iid=False,
                             pre_dispatch=2,
                             verbose=True,
                             )
    estimator.fit(X=1 + np.random.rand(10, 5), y=1. + np.random.rand(10))
    assert estimator.cv_results_['mean_train_score'][0] > -100.0