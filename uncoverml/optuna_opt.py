import json
import logging
import pickle
from pathlib import Path

import optuna
from optuna.distributions import IntDistribution, \
    IntLogUniformDistribution, \
    IntUniformDistribution, \
    FloatDistribution, \
    DiscreteUniformDistribution, \
    CategoricalDistribution


import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, GroupKFold, KFold, cross_validate, GroupShuffleSplit
from sklearn.metrics import check_scoring
from sklearn.svm import SVR
import lightgbm as lgb

#
# "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
# "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
# "num_leaves": trial.suggest_int("num_leaves", 2, 256),
# "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
# "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
# "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
# "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),

# from hyperopt import fmin, tpe, anneal, Trials, space_eval

from uncoverml.config import Config
from uncoverml.optimise.models import transformed_modelmaps as modelmaps
from uncoverml.validate import setup_validation_data
from uncoverml.targets import Targets
from uncoverml import geoio

log = logging.getLogger(__name__)


def optimise_model(X, targets_all: Targets, conf: Config):
    """
    :param X: covaraite matrix
    :param y: targets
    :param w: weights for each target
    :param groups: group number for each target
    :param conf:
    :return:
    """
    # import IPython; IPython.embed(); import sys; sys.exit()
    search_space = {k: eval(v) for k, v in conf.optuna_params_space.items()}
    print(search_space)

    model = modelmaps[conf.algorithm]
    cv_folds = conf.optuna_params.pop('cv') if 'cv' in conf.optuna_params else 5
    random_state = conf.optuna_params.pop('random_state')

    # shuffle data
    rstate = np.random.RandomState(random_state)
    scoring = conf.optuna_params.pop('scoring')
    scorer = check_scoring(model(** conf.algorithm_args), scoring=scoring)

    step = conf.optuna_params.pop('step') if 'step' in conf.optuna_params else 10
    # n_trials = conf.optuna_params.pop('n_trials') if 'n_trials' in conf.optuna_params else 50

    X, y, lon_lat, groups, w, cv = setup_validation_data(X, targets_all, cv_folds, random_state)

    optimizer = optuna.integration.OptunaSearchCV(
        estimator=model(**conf.algorithm_args),
        param_distributions=search_space,
        scoring=scorer,
        cv=cv,
        error_score='raise',
        random_state=rstate,
        return_train_score=True,
        ** conf.optuna_params
    )

    conf.optimised_model = True
    optimizer.fit(X, y, groups=groups, sample_weight=w)
    log.info("Best trial:")
    trial = optimizer.study_.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    save_optimal(random_state, optimizer, conf)


def save_optimal(random_state, optimizer: optuna.integration.OptunaSearchCV, conf: Config):
    reg = modelmaps[conf.algorithm]

    with open(conf.optimised_model_params, 'w') as f:
        if not isinstance(reg(), SVR):
            all_params = {**conf.algorithm_args, 'random_state': random_state}
        else:
            all_params = {**conf.algorithm_args}
        all_params.update(optimizer.best_params_)
        json.dump(all_params, f, sort_keys=True, indent=4, cls=NpEncoder)
        params_str = ''
        for k, v in all_params.items():
            params_str += f"{k}: {v}\n"
        log.info(f"Best params found:\n{params_str}")
        log.info(f"Saved optuna optimised params in {conf.optimised_model_params}")
    results = optimizer.trials_dataframe()
    results.sort_values(by='user_attrs_mean_test_score', ascending=False).to_csv(conf.optimisation_output_optuna)


class NpEncoder(json.JSONEncoder):
    """
    see https://stackoverflow.com/a/57915246/3321542
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
