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

        return cls(ModelSpec.__set_details(mod_dict['training']),
                   ModelSpec.__set_details(mod_dict['prediction']),
                   **mod_dict['parameters'])

    @staticmethod
    def __get_details(func):
        details_dict = {'name': func.__name__,
                        'module': func.__module__
                        }

        if '__self__' in dir(func):
            details_dict['class'] = func.__self__.__class__

        return details_dict

    @staticmethod
    def __set_details(fdict):

        class fun:
            pass

        fun.__name__ = fdict['name']
        fun.__module__ = fdict['module']
        if 'class' in fdict:
            fun.__class__ = fdict['class']

        return fun
