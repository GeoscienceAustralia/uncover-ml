import logging
import click
from os.path import join, abspath
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV

import uncoverml as ls
import uncoverml.config
import uncoverml.features
import uncoverml.geoio
import uncoverml.logging
import uncoverml.targets
from uncoverml.optimise.models import (
    TransformedForestRegressor,
    TransformedGradientBoost,
    )

from uncoverml.config import ConfigException
from uncoverml.transforms import target as transforms

log = logging.getLogger(__name__)

pca = decomposition.PCA()
algos = {
         'randomforest': TransformedForestRegressor(),
         'gradientboost': TransformedGradientBoost(),
         }


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
            param_dict[config.optimisation['algorithm'] + '__' + k] = v

    pipe = Pipeline(steps=steps)
    estimator = GridSearchCV(pipe,
                             param_dict,
                             n_jobs=2,
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
    config.n_subchunks = partitions
    if config.n_subchunks > 1:
        log.info("Memory constraint forcing {} iterations "
                 "through data".format(config.n_subchunks))
    else:
        log.info("Using memory aggressively: dividing all data between nodes")

    # Make the targets
    targets = ls.geoio.load_targets(shapefile=config.target_file,
                                    targetfield=config.target_property)
    # We're doing local models at the moment
    targets_all = ls.targets.gather_targets(targets, node=0)

    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.geoio.image_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    # need to add cubist cols to config.algorithm_args
    x = ls.features.transform_features(image_chunk_sets, transform_sets,
                                       config.final_transform, config)

    x_all = ls.features.gather_features(x, node=0)

    log.info("Optimising {} model".format(config.algorithm))
    estimator.fit(X=x_all, y=targets_all.observations)

    pd.DataFrame.from_dict(
        estimator.cv_results_).sort(columns='rank_test_score').to_csv(
        config.optimisation_output)



