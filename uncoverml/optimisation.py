import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import GroupKFold, KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from uncoverml.config import Config
from uncoverml.optimise.models import transformed_modelmaps as modelmaps
log = logging.getLogger(__name__)


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


regression_metrics = {
    'r2_score': lambda y, py, w:  r2_score(y, py, sample_weight=w),
    'expvar': lambda y, py, w: explained_variance_score(y, py, sample_weight=w),
    'mse': lambda y, py, w: mean_squared_error(y, py, sample_weight=w),
    'mae': lambda y, py, w: mean_absolute_error(y, py, sample_weight=w),
}


def bayesian_optimisation(X, targets_all, conf: Config):
    """
    :param X: covaraite matrix
    :param y: targets
    :param w: weights for each target
    :param groups: group number for each target
    :param conf:
    :return:
    """
    y = targets_all.observations
    groups = targets_all.groups
    w = targets_all.groups
    reg = modelmaps[conf.algorithm](** conf.algorithm_args)
    search_space = {k: eval(v) for k, v in conf.opt_params_space.items()}
    cv_folds = conf.opt_searchcv_params.pop('cv') if 'cv' in conf.opt_searchcv_params else 5
    random_state = conf.opt_searchcv_params['random_state'] if 'random_state' in conf.opt_searchcv_params else None

    if len(np.unique(groups)) >= cv_folds:
        log.info(f'Using GroupKFold with {cv_folds} folds')
        cv = GroupKFold(n_splits=cv_folds)
    else:
        log.info(f'Using KFold with {cv_folds} folds')
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    searchcv = BayesSearchCV(
        reg,
        search_spaces=search_space,
        ** conf.opt_searchcv_params,
        fit_params={'sample_weight': w},
        return_train_score=True,
        refit=False,
        cv=cv
    )

    log.info(f"Optimising params using BayesSearchCV .....")
    searchcv.fit(X, y, groups=groups)

    log.info(f"Finished param optimisation using BayesSearchCV .....")
    log.info(f"Best score found using param optimisation {searchcv.best_score_}")

    with open(conf.optimised_model_params, 'w') as f:
        all_params = {** conf.algorithm_args}
        all_params.update(searchcv.best_params_)
        json.dump(all_params, f, sort_keys=True, indent=4)
        log.info(f"Saved bayesian search optimised params in {all_params}")

    log.info("Now training final model using the optimised model params")
    opt_model = modelmaps[conf.algorithm](** all_params)
    opt_model.fit(X, y, sample_weight=w)
    params = pd.DataFrame.from_dict(searchcv.cv_results_['params'], orient='columns')
    results = pd.DataFrame.from_dict(searchcv.cv_results_)

    df = pd.concat(
        [results, params], axis=1
    ).drop('params', axis=1).sort_values(
        by='rank_test_score', ascending=False
    )

    test_score_col = df.pop('rank_test_score')
    df.insert(
        0, test_score_col.name, test_score_col  # Is in-place
    )
    df.to_csv(
        conf.optimisation_output_skopt
    )
    return opt_model


def score_model(trained_model, X, y, w=None):
    scores = {}
    y_pred = trained_model.predict(X)
    for k, m in regression_metrics.items():
        scores[k] = m(y, y_pred, w)
    return scores