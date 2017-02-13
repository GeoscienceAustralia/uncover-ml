"""
Run the various machine learning parameter optimisation using
scikit-learn GridSearchCv.

.. program-output:: gridsearch --help
"""
import logging
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
from uncoverml.scripts.uncoverml import load_data
from uncoverml.transforms import target as transforms
from uncoverml.optimise.models import transformed_modelmaps as all_modelmaps
log = logging.getLogger(__name__)

pca = decomposition.PCA()
algos = {k: v(ml_score=True) for k, v in all_modelmaps.items()}
algos['transformedgp'] = TransformedGPRegressor(n_restarts_optimizer=10,
                                                normalize_y=True,
                                                ml_score=True)
algos['transformedsvr'] = TransformedSVR(verbose=True, max_iter=1000000,
                                         ml_score=True)


def setup_pipeline(config):
    if config.optimisation['algorithm'] not in algos:
        raise ConfigException('optimisation algo must exist in algos dict')
    steps = []
    param_dict = {}
    if 'featuretransforms' in config.optimisation:
        config.featuretransform = config.optimisation['featuretransforms']
        if 'pca' in config.featuretransform:
            steps.append(('pca', pca))
            for k, v in config.featuretransform['pca'].items():
                param_dict['pca__' + k] = v
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
                             pre_dispatch='2*n_jobs',
                             verbose=True,
                             cv=5,
                             )

    return estimator


@click.command()
@click.argument('pipeline_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
@click.option('-n', '--njobs', type=int, default=-1,
              help='Number of parallel jobs to run. Lower value of n '
                   'reduces memory requirement. '
                   'By default uses all available CPUs')
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def cli(pipeline_file, partitions, njobs, verbosity):
    uncoverml.mllog.configure(verbosity)
    config = ls.config.Config(pipeline_file)
    config.n_jobs = njobs
    estimator = setup_pipeline(config)
    log.info('Running optimisation for {}'.format(
        config.optimisation['algorithm']))

    targets_all, x_all = load_data(config, partitions)

    log.info("Optimising {} model".format(config.optimisation['algorithm']))
    estimator.fit(X=x_all, y=targets_all.observations)

    pd.DataFrame.from_dict(
        estimator.cv_results_).sort_values(by='rank_test_score').to_csv(
        config.optimisation['algorithm'] + '_' + config.optimisation_output)
