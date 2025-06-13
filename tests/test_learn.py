import numpy as np
from types import SimpleNamespace
from uncoverml.learn import local_learn_model
from uncoverml.targets import Targets


def test_local_learn_model():
    config = SimpleNamespace(
        algorithm='randomforest',
        algorithm_args={},
        multicubist=False,
        multirandomforest=False
    )
    x_all = np.random.rand(5, 3)
    targets = Targets(np.random.rand(5, 2), np.random.rand(5), np.zeros(5, dtype=int), np.ones(5))
    targets.fields = ['test']
    model = local_learn_model(x_all, targets, config)
    assert model is not None
    assert hasattr(model, 'predict')


def test_local_learn_model_multirandomforest_path():
    config = SimpleNamespace(
        algorithm='randomforest',
        algorithm_args={},
        multicubist=False,
        multirandomforest=True
    )
    x_all = np.random.rand(6, 4)
    targets = Targets(np.random.rand(6, 2), np.random.rand(6), np.zeros(5, dtype=int), np.ones(6))
    targets.fields = ['test']
    model = local_learn_model(x_all, targets, config)
    assert model is not None
    assert hasattr(model, 'predict')
