import numpy as np

from uncoverml import models


class MLclass:

    def __init__(self):
        self.params = {'p1': 1,
                       'p2': 2,
                       'p3': np.random.randn(1000, 1000)
                       }

    def learn(self):
        pass

    def predict(self):
        pass


def test_ModelSpec():

    def ml_flearn():
        pass

    def ml_fpredict():
        pass

    mcls = MLclass()
    ms_class = models.ModelSpec(mcls.learn, mcls.predict, **mcls.params)
    assert ms_class.train_func == 'learn'
    assert ms_class.predict_func == 'predict'
    assert ms_class.train_module == 'tests.test_models'
    assert ms_class.predict_module == 'tests.test_models'
    assert ms_class.train_class == 'MLclass'
    assert ms_class.predict_class == 'MLclass'
    assert ms_class.params == mcls.params
    assert ms_class.to_dict() == \
        models.ModelSpec.from_dict(ms_class.to_dict()).to_dict()

    ms_func = models.ModelSpec(ml_flearn, ml_fpredict, **mcls.params)
    assert ms_func.train_func == 'ml_flearn'
    assert ms_func.predict_func == 'ml_fpredict'
    assert ms_func.train_module == 'tests.test_models'
    assert ms_func.predict_module == 'tests.test_models'
    assert ms_func.params == mcls.params
    assert ms_func.to_dict() == \
        models.ModelSpec.from_dict(ms_func.to_dict()).to_dict()
