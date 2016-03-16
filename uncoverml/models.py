""" Model Spec Objects and ML algorithm serialisation. """


class ModelSpec:

    def __init__(self, train_func, pred_func, **params):

        self.train_details = ModelSpec.__get_details(train_func)
        self.pred_details = ModelSpec.__get_details(pred_func)
        self.params = params

    @property
    def train_func(self):
        return self.train_details['name']

    @property
    def train_module(self):
        return self.train_details['module']

    @property
    def train_class(self):
        return self.train_details['class'] \
            if 'class' in self.train_details else None

    @property
    def predict_func(self):
        return self.pred_details['name']

    @property
    def predict_module(self):
        return self.pred_details['module']

    @property
    def predict_class(self):
        return self.pred_details['class'] \
            if 'class' in self.pred_details else None

    def to_dict(self):
        return {'training': self.train_details,
                'prediction': self.pred_details,
                'parameters': self.params
                }

    @classmethod
    def from_dict(cls, mod_dict):

        # dummy construct
        def dummy():
            pass

        retcls = cls(dummy, dummy, **{})

        # overwrite real properties
        retcls.train_details = mod_dict['training']
        retcls.pred_details = mod_dict['prediction']
        retcls.params = mod_dict['parameters']

        return retcls

    @staticmethod
    def __get_details(func):
        details_dict = {'name': func.__name__,
                        'module': func.__module__
                        }

        if '__self__' in dir(func):
            details_dict['class'] = func.__self__.__class__.__name__

        return details_dict
