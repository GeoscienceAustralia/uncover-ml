"""
Run the various machine learning parameter optimisation using
scikit-learn GridSearchCv.

Available scorers:
    Regression:
        'r2', 'expvar', 'smse', 'lins_ccc'
    Classification:
        'accuracy'
    Classification with probability:
        'log_loss', 'auc'

Not yet implemented:
    Regression with variance (models w/ `predict_dist`):
        'mll'
    Classification:
        'mean_confusion', 'mean_confusion_normalized'

If no scorers are provided, then the default `score` method of the model
will be used.

Note that GridSearchCV is run with `refit` set to False. This is because
the `_best_estimator` (best model found by GridSearchCV) is never used
to predict anything, so fitting it is a waste.

If you ever do want to use refit, keep in mind that if you use mutliple
scorers, refit needs to be set to the name of a scorer used to find the
best parameters for refitting the model.

.. program-output:: gridsearch --help
"""
import logging
import os
from collections import OrderedDict
from itertools import product

import click
import pandas as pd
from sklearn import decomposition
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import uncoverml as ls
import uncoverml.config
import uncoverml.mllog
from uncoverml.config import ConfigException
from uncoverml.optimise.models import (
    TransformedGPRegressor,
    kernels,
    TransformedSVR
    )
from uncoverml.scripts.learn_cli import _load_data
from uncoverml.transforms import target as transforms
from uncoverml.optimise.models import transformed_modelmaps as all_modelmaps
from uncoverml.optimise.scorers import (regression_predict_scorers, classification_predict_scorers,
                                        classification_proba_scorers)

_logger = logging.getLogger(__name__)


pca = decomposition.PCA()
algos = {k: v() for k, v in all_modelmaps.items()}
algos['transformedgp'] = TransformedGPRegressor(n_restarts_optimizer=10,
                                                normalize_y=True)
algos['transformedsvr'] = TransformedSVR(verbose=True, max_iter=1000000)


def setup_pipeline(config):
    if config.optimisation['algorithm'] not in algos:
        raise ConfigException('Optimisation algorithm must exist in avilable algorithms: {}'
                              .format(list(algos.keys())))

    steps = []
    param_dict = {}

    if 'featuretransforms' in config.optimisation:
        config.featuretransform = config.optimisation['featuretransforms']
        if 'pca' in config.featuretransform:
            steps.append(('pca', pca))
            for k, v in config.featuretransform['pca'].items():
                param_dict['pca__' + k] = v

    if 'scorers' in config.optimisation:
        scorers = config.optimisation['scorers']
        scorer_maps = [regression_predict_scorers, classification_proba_scorers, 
                       classification_predict_scorers]

        scoring = {}

        for s in scorers:
            for sm in scorer_maps:
                f = sm.get(s)
                if f is not None:
                    break
            if f is None:
                _logger.warning(f"Scorer '{s}' not found!")
            else:
                scoring[s] = f
        if not scoring:
            scoring = None
    else:
        scoring = None

    if 'hyperparameters' in config.optimisation:
        steps.append((config.optimisation['algorithm'],
                      algos[config.optimisation['algorithm']]))
        for k, v in config.optimisation['hyperparameters'].items():
            if k == 'target_transform':
                v = [transforms.transforms[vv]() for vv in v]
            if k == 'kernel':
                # for scikitlearn kernels
                if isinstance(v, dict):
                    V = []
                    for kk, value in v.items():
                        value = OrderedDict(value)
                        values = [v for v in value.values()]
                        prod = product(* values)
                        keys = value.keys()
                        combinations = []
                        for p in prod:
                            d = {}
                            for kkk, pp in zip(keys, p):
                                d[kkk] = pp
                            combinations.append(d)
                        V += [kernels[kk](** c) + WhiteKernel()
                              for c in combinations]
                    v = V
                    
            param_dict[config.optimisation['algorithm'] + '__' + k] = v

    pipe = Pipeline(steps=steps)

    estimator = GridSearchCV(pipe,
                             param_dict,
                             n_jobs=config.n_jobs,
                             iid=False,
                             scoring=scoring,
                             refit=False,
                             pre_dispatch='2*n_jobs',
                             verbose=True,
                             cv=5,
                             )

    return estimator, scoring


def main(pipeline_file, partitions, njobs):
    config = ls.config.Config(pipeline_file)
    config.n_jobs = njobs
    estimator, scoring = setup_pipeline(config)
    _logger.info('Running optimisation for {}'.format(config.optimisation['algorithm']))

    training_data, _ = _load_data(config, partitions)
    targets_all = training_data.targets_all
    x_all = training_data.x_all
    
    _logger.info("Optimising {} model".format(config.optimisation['algorithm']))
    # Runs 'fit' on selected model ('estimator' in scikit-learn) with 
    # hyperparameter combinations.
    estimator.fit(X=x_all, y=targets_all.observations)

    if scoring is None:
        sort_by = 'rank_test_score'
    else:
        sort_by = 'rank_test_' + list(scoring.keys())[0]

    pd.DataFrame.from_dict(estimator.cv_results_).sort_values(by=sort_by)\
      .to_csv(config.optimisation_results_file)

