import logging
import click
from os.path import join
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import uncoverml as ls
import uncoverml.config
import uncoverml.features
import uncoverml.geoio
import uncoverml.logging
import uncoverml.targets
from uncoverml.models import TransformedForestRegressor

log = logging.getLogger(__name__)

pca = decomposition.PCA()
n_components = [2, 5]
n_estimators = [2, 20, 50, 200]
alphas = [0.01, 0.05, 0.1]
beta1s = [0.1, 0.5, 0.9]
fit_intercept = [True, False]
outdir = ['.']
target_transform = ['identity', 'log']
criterion = ['mse', 'mae']

# algos = {'randomforest': RandomForestRegressor,
#          'randomforesttransformed': RandomForestTransformed,
#          'gradiendboost': GradientBoostingRegressor,
#          'transformedforest': TransformedForestRegressor,
#          }


def setup_rf_transformed(config):
    pipe = Pipeline(steps=[('pca', pca),
                           # ('randomforest',
                           #  RandomForestRegressor()),
                           ('randomforesttransformed',
                            TransformedForestRegressor()),
                           ])
    estimator = GridSearchCV(
        pipe,
        dict(
            randomforesttransformed__n_estimators=n_estimators,
            randomforesttransformed__target_transform=target_transform,
            ),
        n_jobs=2,
        iid=False)

    return estimator


def setup_rf(config):
    pipe = Pipeline(steps=[('pca', pca),
                           ('randomforest',
                            RandomForestRegressor()),
                           ])
    estimator = GridSearchCV(
        pipe,
        dict(
            randomforest__n_estimators=n_estimators,
            randomforest__criterion=criterion,
            ),
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

    estimator = setup_rf_transformed(config)
    # estimator = setup_rf(config)
    log.info('Running optimisation for {}'.format(config.algorithm))
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
    pca.fit(x_all)
    estimator.fit(X=x_all, y=targets_all.observations)

    # print(estimator.cv_results_)
    # print('='*20)
    # print(estimator.best_estimator_)
    # print('=' * 20)
    # print(estimator.cv)
    # print('=' * 20)
    # print(estimator.best_score_)
    # print('=' * 20)
    # print(estimator.best_params_)
    # print('=' * 20)
    pd.DataFrame.from_dict(estimator.cv_results_).to_csv(
        join(config.output_dir, 'test.csv'))



