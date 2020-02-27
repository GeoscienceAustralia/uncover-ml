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
    targets_all, x_all = _load_data(config, partitions)

    if config.cross_validate:
        run_crossval(x_all, targets_all, config)

    _logger.info("Learning full {} model".format(config.algorithm))
    model = ls.learn.local_learn_model(x_all, targets_all, config)

    # use trained model
    if config.permutation_importance:
        ls.mpiops.run_once(ls.validate.permutation_importance, model, x_all,
                           targets_all, config)

    ls.mpiops.run_once(ls.geoio.export_model, model, config)
    if config.crop_box:
        ls.mpiops.run_once(_clean_temp_cropfiles, config)

    _logger.info("Finished! Total mem = {:.1f} GB".format(_total_gb()))


def run_crossval(x_all, targets_all, config):
    crossval_results = ls.validate.local_crossval(x_all,
                                                  targets_all, config)
    if ls.mpiops.chunk_index == 0:
        crossval_results.export_crossval(config)


def _load_data(config, partitions):
    if config.pk_load:
        x_all = pickle.load(open(config.pk_covariates, 'rb'))
        targets_all = pickle.load(open(config.pk_targets, 'rb'))
        if config.cubist or config.multicubist:
            config.algorithm_args['feature_type'] = \
                pickle.load(open(config.pk_featurevec, 'rb'))
        _logger.warning("Using  pickled targets and covariates. Make sure you have"
                        " not changed targets file and/or covariates.")
    else:
        if config.crop_box:
            ls.geoio.crop_covariates(config)
        config.n_subchunks = partitions
        if config.n_subchunks > 1:
            _logger.info("Memory constraint forcing {} iterations "
                         "through data".format(config.n_subchunks))
        else:
            _logger.info("Using memory aggressively: "
                         "dividing all data between nodes")

        # Make the targets
        _logger.info("Intersecting targets as pickled train data was not "
                     "available")
        targets = ls.geoio.load_targets(shapefile=config.target_file,
                                        targetfield=config.target_property,
                                        covariate_crs=ls.geoio.get_image_crs(config),
                                        crop_box=config.crop_box)
                                            
        # Get the image chunks and their associated transforms
        image_chunk_sets = ls.geoio.image_feature_sets(targets, config)
        transform_sets = [k.transform_set for k in config.feature_sets]

        if config.raw_covariates:
            _logger.info("Saving raw data before any processing")
            ls.features.save_intersected_features_and_targets(image_chunk_sets,
                                                              transform_sets, targets, config)

        if config.rank_features:
            _logger.info("Ranking features...")
            measures, features, scores = \
                ls.validate.local_rank_features(image_chunk_sets, transform_sets, targets, config)
            ls.mpiops.run_once(ls.geoio.export_feature_ranks, measures, features, scores, config)

        # need to add cubist cols to config.algorithm_args
        # keep: bool array corresponding to rows that are retained
        features, keep = ls.features.transform_features(image_chunk_sets,
                                                        transform_sets,
                                                        config.final_transform,
                                                        config)
        # learn the model
        # local models need all data
        x_all = ls.features.gather_features(features[keep], node=0)

        # We're doing local models at the moment
        targets_all = ls.targets.gather_targets(targets, keep, config, node=0)

        # Pickle data if requested.
        if ls.mpiops.chunk_index == 0:
            if config.pk_covariates and not os.path.exists(config.pk_covariates):
                pickle.dump(x_all, open(config.pk_covariates, 'wb'))
            if config.pk_targets and not os.path.exists(config.pk_targets):
                pickle.dump(targets_all, open(config.pk_targets, 'wb'))

    return targets_all, x_all

def _total_gb():
    # given in KB so convert
    my_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    # total_usage = mpiops.comm.reduce(my_usage, root=0)
    total_usage = ls.mpiops.comm.allreduce(my_usage)
    return total_usage

def _clean_temp_cropfiles(config):
    shutil.rmtree(config.tmpdir)   

