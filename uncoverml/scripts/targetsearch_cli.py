"""
Run the uncoverml pipeline for clustering, supervised learning and prediction.

.. program-output:: uncoverml --help
"""
import logging
import pickle
import resource
from os.path import isfile, splitext, exists
import os
import shutil
import warnings
import sys

import click
import numpy as np
import matplotlib
matplotlib.use('Agg')

import uncoverml as ls
import uncoverml.cluster
import uncoverml.config
import uncoverml.features
import uncoverml.geoio
import uncoverml.learn
import uncoverml.mllog
import uncoverml.mpiops
import uncoverml.predict
import uncoverml.validate
import uncoverml.targets
import uncoverml.models
from uncoverml.transforms import StandardiseTransform


_logger = logging.getLogger(__name__)
# warnings.showwarning = warn_with_traceback
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def main(config_file, partitions):
    config = ls.config.Config(config_file)
    if config.crop_box:
        ls.geoio.crop_covariates(config)
    config.n_subchunks = partitions
    if config.n_subchunks > 1:
        _logger.info("Memory constraint forcing {} iterations "
                     "through data".format(config.n_subchunks))
    else:
        _logger.info("Using memory aggressively: "
                             "dividing all data between nodes")

    _logger.info("Loading real targets and generating training targets...")

    # Load all targets. These will be labelled 'query'.
    # TODO: Drop targets from prediction area?
    query_targets = ls.geoio.load_targets(shapefile=config.target_file,
                                        targetfield=config.target_property,
                                        covariate_crs=ls.geoio.get_image_crs(config),
                                        crop_box=config.crop_box)
    query_targets = ls.targets.label_targets(query_targets, 'query')

    # Get random sample of targets from within prediction area. These 
    #  will be labelled 'training'.
    training_targets = ls.targets.generate_dummy_targets(config.target_crop_box, 'training', 
            query_targets.positions.shape[0])
    training_targets = ls.targets.label_targets(training_targets, 'training')
    # Write them out for debug purposes.
    ls.targets.save_target_positions(training_targets, config.target_search_points)

    _logger.debug(f"Class 1: {query_targets.positions.shape}, "
                  f"'{query_targets.observations[0]}' and class 2: "
                  f"{training_targets.positions.shape}, '{training_targets.observations[0]}'")

    targets = ls.targets.merge_targets(query_targets, training_targets)


    image_chunk_sets = ls.geoio.image_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]
    features, keep = ls.features.transform_features(image_chunk_sets,
                                                    transform_sets,
                                                    config.final_transform,
                                                    config)
    x_all = ls.features.gather_features(features[keep], node=0)
    targets_all = ls.targets.gather_targets(targets, keep, config, node=0)
    model = ls.models.LogisticClassifier(random_state=1)
    ls.models.apply_multiple_masked(model.fit, (x_all, targets_all.observations),
                                    kwargs={'fields': targets_all.fields,
                                            'lon_lat': targets_all.positions})

    predict_tags = model.get_predict_tags()
    _logger.debug(f"Predict tags: {predict_tags}")

    for i in range(config.n_subchunks):
        _logger.info("starting to precit partition {}".format(i+1))
        x, _ = ls.predict._get_data(i, config)
        y_star = ls.predict.predict(x, model, config.quantiles)

    y_star = ls.mpiops.comm.gather(y_star, root=0)
    ls.mpiops.comm.barrier()
    if ls.mpiops.chunk_index == 0:
        y_star_all = np.concatenate(y_star)
        _logger.debug(f"y_star_all: {y_star_all.T.shape}")
        _logger.debug(f"{y_star_all.T[1]}")
        # Out 
    raise Exception("DIE")

    if config.crop_box:
        ls.mpiops.run_once(_clean_temp_cropfiles, config)


def _clean_temp_cropfiles(config):
    shutil.rmtree(config.tmpdir)   

