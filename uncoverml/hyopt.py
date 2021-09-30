import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GroupKFold, KFold, cross_validate
from hyperopt import fmin, tpe, anneal, Trials
from hyperopt.hp import uniform, randint, choice, loguniform, quniform
from uncoverml.config import Config
from uncoverml.optimise.models import transformed_modelmaps as modelmaps
from uncoverml import geoio

log = logging.getLogger(__name__)


hp_algo = {
    'bayes': tpe.suggest,
    'anneal': anneal.suggest
}


def optimise_model(X, targets_all, conf: Config):
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
    trials = Trials()
    search_space = {k: eval(v) for k, v in conf.hp_params_space.items()}

    reg = modelmaps[conf.algorithm]

    algo = hp_algo[conf.hyperopt_params.pop('algo')] if 'algo' in conf.hyperopt_params else tpe.suggest
    cv_folds = conf.hyperopt_params.pop('cv') if 'cv' in conf.hyperopt_params else 5
    random_state = conf.hyperopt_params.pop('random_state')
    rstate = np.random.RandomState(random_state)
    scoring = conf.hyperopt_params.pop('scoring')

    if len(np.unique(groups)) >= cv_folds:
        log.info(f'Using GroupKFold with {cv_folds} folds')
        cv = GroupKFold(n_splits=cv_folds)
    else:
        log.info(f'Using KFold with {cv_folds} folds')
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(params, random_state=random_state, cv=cv, X=X, y=y):
        # the function gets a set of variable parameters in "param"
        all_params = {**conf.algorithm_args}

        if hasattr(reg, 'random_state'):
            all_params.update(**params, random_state=random_state)
            model = reg(** all_params)
        else:
            all_params.update(** params)
            model = reg(** all_params)
        print("="*50)
        log.info(f"Cross-validating param combination:\n {all_params}")
        # and then conduct the cross validation with the same folds as before
        cv_score = cross_val_score(model, X, y,
                                   fit_params={'sample_weight': w},
                                   groups=groups, cv=cv, scoring=scoring, n_jobs=-1).mean()
        score = 1 - cv_score
        log.info(f"Loss: {score}")
        return score

    step = conf.hyperopt_params.pop('step') if 'step' in conf.hyperopt_params else 10
    max_evals = conf.hyperopt_params.pop('max_evals') if 'max_evals' in conf.hyperopt_params else 50

    log.info(f"Optimising params using Hyperopt {algo}")

    for i in range(1, max_evals + 1, step):
        # fmin runs until the trials object has max_evals elements in it, so it can do evaluations in chunks like this
        best = fmin(
            objective, search_space,
            ** conf.hyperopt_params,
            algo=algo,
            trials=trials,
            max_evals=i + step,
            rstate=rstate
            )
        # each step 'best' will be the best trial so far
        log.info(f"After {i + step} trials best config: \n {best}")
        # each step 'trials' will be updated to contain every result
        # you can save it to reload later in case of a crash, or you decide to kill the script
        pickle.dump(trials, open(Path(conf.output_dir).joinpath(f"hpopt_{i + step}.pkl"), "wb"))
        save_optimal(best, trials, objective, conf)

    log.info(f"Finished param optimisation using Hyperopt {algo}")
    log.info(f"Best score found using param optimisation {objective(best)}")

    with open(conf.optimised_model_params, 'w') as f:
        all_params = {** conf.algorithm_args}
        all_params.update(best)
        # json.dump(all_params, cls=NpEncoder)
        json.dump(all_params, f, sort_keys=True, indent=4, cls=NpEncoder)
        log.info(f"Saved {algo} optimised params in {all_params}")

    log.info("Now training final model using the optimised model params")
    opt_model = modelmaps[conf.algorithm](** all_params)
    opt_model.fit(X, y, sample_weight=w)

    conf.optimised_model = True
    geoio.export_model(opt_model, conf, False)


def save_optimal(best, trials, objective, conf: Config):
    params_space = []
    for t in trials.trials:
        l = {k: v[0] for k, v in t['misc']['vals'].items()}
        params_space.append(l)
    results = pd.DataFrame.from_dict(params_space, orient='columns')
    loss = [x['result']['loss'] for x in trials.trials]
    results.insert(0, 'loss', loss)
    log.info("Best Loss {:.3f} params {}".format(objective(best), best))
    results.sort_values(by='loss').to_csv(conf.optimisation_output_hpopt)


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
