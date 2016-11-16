import logging
import click
import pandas as pd
from collections import OrderedDict

from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import (
    WhiteKernel,
    ConstantKernel,
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
)

import uncoverml as ls
import uncoverml.config
import uncoverml.logging
from uncoverml.scripts.uncoverml import load_data
from uncoverml.optimise.models import (
    TransformedForestRegressor,
    TransformedGradientBoost,
    TransformedGPRegressor,
)
from uncoverml.config import ConfigException
from uncoverml.transforms import target as transforms

log = logging.getLogger(__name__)

pca = decomposition.PCA()
algos = {'randomforest': TransformedForestRegressor(),
         'gradientboost': TransformedGradientBoost(),
         'gp': TransformedGPRegressor(n_restarts_optimizer=3,
                                      normalize_y=True),
         }

kernels = {'rbf': RBF,
           'matern': Matern,
           'quadratic': RationalQuadratic,
           'expsinesqauared': ExpSineSquared,
           }
LENGTH_SCALE = 'length_scale'
NU = 'nu'


def setup_pipeline(config):
    if config.optimisation['algorithm'] not in algos:
        raise ConfigException('optimisation algo must exist in algos dict')
    steps = []
    param_dict = {}
    from itertools import product
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
                             n_jobs=-1,
                             iid=False
                             )

    return estimator


@click.group()
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def cli(verbosity):
    ls.logging.configure(verbosity)


@cli.command()
@click.argument('pipeline_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
def optimise(pipeline_file, partitions):
    config = ls.config.Config(pipeline_file)
    estimator = setup_pipeline(config)
    log.info('Running optimisation for {}'.format(
        config.optimisation['algorithm']))

    targets_all, x_all = load_data(config, partitions)

    log.info("Optimising {} model".format(config.optimisation['algorithm']))
    estimator.fit(X=x_all, y=targets_all.observations)

    pd.DataFrame.from_dict(
        estimator.cv_results_).sort_values(by='rank_test_score').to_csv(
        config.optimisation_output)
