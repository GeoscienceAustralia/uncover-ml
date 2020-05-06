"""
Wrappers around model scoring metrics so they can be used in 
GridSearchCV.
"""
import numpy as np
from sklearn.metrics import (explained_variance_score, r2_score,
                             accuracy_score, log_loss, roc_auc_score,
                             confusion_matrix)
from sklearn.metrics import make_scorer
from sklearn.metrics._scorer import _BaseScorer
from revrand.metrics import lins_ccc, mll, smse

from uncoverml.validate import _binarizer

# lower is better: mll, smse
lower_is_better = ['mll', 'smse']

# Any scorer with that takes a `predict` or `predict_proba` output
# can be wrapped with sklearn's `make_scorer`.

# Regression
r2_scorer = make_scorer(r2_score)
expvar_scorer = make_scorer(explained_variance_score)
smse_scorer = make_scorer(smse, greater_is_better=False)
lins_ccc_scorer = make_scorer(lins_ccc)

regression_predict_scorers = {
    'r2': r2_scorer,
    'expvar': expvar_scorer,
    'smse': smse_scorer,
    'lins_ccc': lins_ccc_scorer,
}

# Classification
accuracy_scorer = make_scorer(accuracy_score)

# TODO: scorers that output a collection need to be wrapped into multiple individual scorers that 
# each return a scalar.
# mean_confusion = make_scorer(lambda y, ey: (confusion_matrix(y, ey)).tolist())
# mean_confusion_normalized = make_scorer(lambda y, ey: (confusion_matrix(y, ey) / len(y)).tolist())

classification_predict_scorers = {
    'accuracy': accuracy_scorer,
}

log_loss_scorer = make_scorer(log_loss, needs_proba=True)
auc_scorer = make_scorer(lambda y, p: _binarizer(y, p, roc_auc_score, average='macro'), needs_proba=True)

classification_proba_scorers = {
    'log_loss': log_loss_scorer,
    'auc': auc_scorer
}

# Variance metrics are not yet implemented: we need to subclass 
# sklearn.Pipeline to include a 'predict_dist' method that calls the
# underlying model's 'predict_dist'

# Scorers that require output from 'predict_dist' need to be subclassed
# before being wrapped.
# `sign = -1` is equivalent to `greater_is_better = False` in `make_scorer`
class _VarianceScorer(_BaseScorer):
    """
    See sklearn's _PredictScorer:
    https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/metrics/_scorer.py#L176
    """
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        Ey, Vy, _, _ = method_caller(estimator, 'predict_dist', X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, Ey, Vy,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, Ey, Vy, **self.kwargs)

mll_scorer = _VarianceScorer(mll, sign=-1, kwargs={})

regression_variance_scorers = {
    'mll': mll_scorer
}

# Transformed metrics not yet implemented: need to subclass 
# sklearn.Pipeline and provide a 'target_transform' method. 

# Note: sklearn Pipeline handles transforms in its own way, so these
# may not even be needed.

# Transformed models need subclasses for their scorers so we can compute
# score on transformed y_true and y_pred.
class _TransformedPredictScorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        y_pred = method_caller(estimator, 'predict', X)
        yt = estimator.target_transform.transform(y_true)
        pyt = estimator.target_transform.transform(y_pred)
        if sample_weight is not None:
            return self._sign * self._score_func(yt, pyt,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(yt, pyt, **self.kwargs)

transformed_regression_predict_scorers = {}
for k, v in regression_predict_scorers.items():
    if k in lower_is_better:
        sign = -1
    else:
        sign = 1
    transformed_regression_predict_scorers[k] = _TransformedPredictScorer(v._score_func, sign, kwargs={})

class _TransformedVarianceScorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        Ey, Vy, _, _ = method_caller(estimator, 'predict_dist', X)
        yt = estimator.target_transform.transform(y_true)
        eyt = estimator.target_transform.transform(Ey)
        vyt = np.square(estimator.target_transform.transform(np.sqrt(Vy)))
        if sample_weight is not None:
            return self._sign * self._score_func(yt, eyt, vyt,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(yt, eyt, vyt, **self.kwargs)

transformed_regression_variance_scorers = {}
for k, v in regression_variance_scorers.items():
    if k in lower_is_better:
        sign = -1
    else:
        sign = 1
    transformed_regression_variance_scorers[k] = _TransformedVarianceScorer(v._score_func, sign, kwargs={})

