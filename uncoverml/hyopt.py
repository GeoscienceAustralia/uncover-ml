import sys
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, GroupKFold, KFold, cross_validate, GroupShuffleSplit
from sklearn.metrics import check_scoring
from sklearn.svm import SVR
from hyperopt import fmin, tpe, anneal, Trials, space_eval

from hyperopt.hp import uniform, randint, choice, loguniform, quniform
from uncoverml.config import Config
from uncoverml.optimise.models import transformed_modelmaps as modelmaps
from uncoverml.validate import setup_validation_data
from uncoverml.targets import Targets
from uncoverml import geoio
from uncoverml.log_progress import write_progress_to_file

log = logging.getLogger(__name__)


hp_algo = {
    'bayes': tpe.suggest,
    'anneal': anneal.suggest
}


def optimise_model(X, targets_all: Targets, conf: Config):
    """
    :param X: covaraite matrix
    :param y: targets
    :param w: weights for each target
    :param groups: group number for each target
    :param conf:
    :return:
    """
    y = targets_all.observations
    lon_lat = targets_all.positions
    groups = targets_all.groups
    w = targets_all.groups
    trials = Trials()
    search_space = {k: eval(v) for k, v in conf.hp_params_space.items()}

    reg = modelmaps[conf.algorithm]
    has_random_state_arg = hasattr(reg(), 'random_state')
    bayes_or_anneal = conf.hyperopt_params.pop('algo') if 'algo' in conf.hyperopt_params else 'bayes'
    algo = hp_algo[bayes_or_anneal]
    cv_folds = conf.hyperopt_params.pop('cv') if 'cv' in conf.hyperopt_params else 5
    random_state = conf.hyperopt_params.pop('random_state')
    conf.bayes_or_anneal = bayes_or_anneal
    conf.algo = algo

    # shuffle data
    rstate = np.random.RandomState(random_state)
    scoring = conf.hyperopt_params.pop('scoring')
    scorer = check_scoring(reg(** conf.algorithm_args), scoring=scoring)

    write_progress_to_file('opt', 'Setting up optimisation', conf)
    X, y, lon_lat, groups, w, cv = setup_validation_data(X, targets_all, cv_folds, random_state)

    def objective(params, random_state=random_state, cv=cv, X=X, y=y):
        # the function gets a set of variable parameters in "param"
        all_params = {**conf.algorithm_args}
        if has_random_state_arg and (not isinstance(reg(), SVR)):
            all_params.update(**params, random_state=random_state)
            model = reg(** all_params)
        else:
            all_params.update(** params)
            model = reg(** all_params)
        print("="*50)
        params_str = ''
        for k, v in all_params.items():
            params_str += f"{k}: {v}\n"
        log.info(f"Cross-validating param combination:\n{params_str}")
        cv_results = cross_validate(model, X, y,
                                    fit_params={'sample_weight': w},
                                    groups=groups, cv=cv, scoring={'score': scorer}, n_jobs=-1)
        score = 1 - cv_results['test_score'].mean()
        log.info(f"Loss: {score}")
        return score

    step = conf.hyperopt_params.pop('step') if 'step' in conf.hyperopt_params else 10
    max_evals = conf.hyperopt_params.pop('max_evals') if 'max_evals' in conf.hyperopt_params else 50

    log.info(f"Optimising params using Hyperopt {algo}")
    write_progress_to_file('opt', f'Begin pptimising params using Hyperopt {algo}', conf)
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
        params_str = ''
        best = space_eval(search_space, best)
        for k, v in best.items():
            params_str += f"{k}: {v}\n"
        log.info(f"After {i + step} trials best config: \n {params_str}")
        # each step 'trials' will be updated to contain every result
        # you can save it to reload later in case of a crash, or you decide to kill the script
        pickle.dump(trials, open(Path(conf.output_dir).joinpath(f"hpopt_{i + step}.pkl"), "wb"))
        save_optimal(best, random_state, trials, objective, conf)
        opt_progress = float(i)/float(max_evals+1)
        write_progress_to_file('opt', f'Optimisation progress: {opt_progress:.2%}', conf)

    write_progress_to_file('opt', 'Finished optimisation, now training optimised model', conf)
    log.info(f"Finished param optimisation using Hyperopt")
    all_params = {** conf.algorithm_args}
    all_params.update(best)
    log.info("Now training final model using the optimised model params")
    opt_model = modelmaps[conf.algorithm](** all_params)

    progress_file = Path(conf.output_dir) / 'opt_progress.txt'
    sys.stdout = open(str(progress_file), 'a')
    opt_model.fit(X, y, sample_weight=w)
    sys.stdout.close()
    write_progress_to_file('opt', 'Optimised model trained, now export', conf)

    conf.optimised_model = True
    geoio.export_model(opt_model, conf, False)
    write_progress_to_file('opt', 'Optimisation complete, model exported', conf)


def save_optimal(best, random_state, trials, objective, conf: Config):
    reg = modelmaps[conf.algorithm]

    with open(conf.optimised_model_params, 'w') as f:
        if not isinstance(reg(), SVR):
            all_params = {**conf.algorithm_args, 'random_state': random_state}
        else:
            all_params = {**conf.algorithm_args}
        all_params.update(best)
        # json.dump(all_params, cls=NpEncoder)
        json.dump(all_params, f, sort_keys=True, indent=4, cls=NpEncoder)
        params_str = ''
        for k, v in all_params.items():
            params_str += f"{k}: {v}\n"
        log.info(f"Best params found:\n{params_str}")
        log.info(f"Saved hyperopt.{conf.bayes_or_anneal}.{conf.algo.__name__} "
                 f"optimised params in {conf.optimised_model_params}")

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
