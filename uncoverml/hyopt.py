import logging

import numpy as np
from sklearn.model_selection import cross_val_score, GroupKFold, KFold
from xgboost.sklearn import XGBRegressor
from hyperopt import fmin, tpe, anneal, Trials
from hyperopt.hp import uniform, randint, choice, loguniform, quniform
from uncoverml.config import Config
from uncoverml.optimise.models import transformed_modelmaps as modelmaps

log = logging.getLogger(__name__)

# space={'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),
#        'max_depth' : hp.quniform('max_depth', 2, 20, 1),
#        'learning_rate': hp.loguniform('learning_rate', -5, 0)
#        }

hp_algo = {
    'bayes': tpe.suggest,
    'anneal': anneal.suggest
}



def bayesian_optimisation(X, y, w, groups, conf: Config):
    """
    :param X: covaraite matrix
    :param y: targets
    :param w: weights for each target
    :param groups: group number for each target
    :param conf:
    :return:
    """
    trials = Trials()
    search_space = {k: eval(v) for k, v in conf.hp_params_space.items()}
    # search_space={'n_estimators': quniform('n_estimators', 10, 50, 1),
    #        'max_depth' : quniform('max_depth', 2, 20, 1),
    #        'learning_rate': loguniform('learning_rate', -5, 0)
    #        }

    reg = modelmaps[conf.algorithm]

    algo = hp_algo[conf.hyperopt_params.pop('algo')] if 'algo' in conf.hyperopt_params else tpe.suggest
    print(algo)
    print(search_space)
    cv_folds = conf.hyperopt_params.pop('cv') if 'cv' in conf.hyperopt_params else 5
    random_state = conf.hyperopt_params.pop('rstate')
    rstate = np.random.RandomState(random_state)

    if len(np.unique(groups)) > 1:
        log.info(f'Using GroupKFold with {cv_folds} folds')
        cv = GroupKFold(n_splits=cv_folds)
    else:
        log.info(f'Using KFold with {cv_folds} folds')
        cv = KFold(n_splits=cv_folds)

    # def gb_mse_cv(params, X=X, y=y, groups=groups):
    #     # the function gets a set of variable parameters in "param"
    #     # params = {'n_estimators': int(params['n_estimators']),
    #     #           'max_depth': int(params['max_depth']),
    #     #           'learning_rate': params['learning_rate']}
    #
    #     # we use this params to create a new LGBM Regressor
    #     all_params = {**conf.algorithm_args}
    #     print(params)
    #     if hasattr(reg, 'random_state'):
    #         all_params.update(**params, random_state=random_state)
    #         model = reg(** all_params)
    #     else:
    #         all_params.update(** params)
    #         model = reg(** all_params)
    #
    #     # and then conduct the cross validation with the same folds as before
    #     score = -cross_val_score(model, X, y, groups=groups,
    #                              cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()
    #
    #     return score
    def gb_mse_cv(params, random_state=random_state, cv=cv, X=X, y=y):
        # the function gets a set of variable parameters in "param"
        all_params = {**conf.algorithm_args}

        if hasattr(reg, 'random_state'):
            all_params.update(**params, random_state=random_state)
            model = reg(** params)
        else:
            all_params.update(** params)
            model = reg(** params)

        print(conf.algorithm_args)
        print(params)
        print(all_params)
        print("=" * 50)

        # we use this params to create a new LGBM Regressor
        # model = reg(random_state=random_state, **params)

        # and then conduct the cross validation with the same folds as before
        score = -cross_val_score(model, X, y,
                                 fit_params={'sample_weight': w},
                                 groups=groups, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()

        return score

    import IPython; IPython.embed(); import sys; sys.exit()
    best = fmin(fn=gb_mse_cv,  # function to optimize
                space=search_space,
                ** conf.hyperopt_params,
                algo=algo,  # optimization algorithm, hyperotp will select its parameters automatically
                trials=trials,  # logging
                rstate=rstate,
                )


    model = modelmaps[conf.algorithm]()
    model.fit(train_data, train_targets)
    tpe_test_score = mean_squared_error(test_targets, model.predict(test_data))

    print("Best MSE {:.3f} params {}".format(gb_mse_cv(best), best))

    # reg = modelmaps[conf.algorithm](** conf.algorithm_args)
    #
    # estim = HyperoptEstimator(
    #     preprocessing=[IdentityTransformer],
    #     # space=,
    #     regressor=xgboost_regression('xgb', objective="reg:squarederror"),
    #     max_evals=20,
    #     trial_timeout=60,
    #     seed=1,
    #     verbose=True,
    # )
    # estim.fit(X, y, groups=groups, n_folds=2)
    # estim.retrain_best_model_on_full_data(X, y)
    # print('Best preprocessing pipeline:')
    # for pp in estim._best_preprocs:
    #     print(pp)
    # print('\n')
    # print('Best regressor:\n', estim._best_learner)

