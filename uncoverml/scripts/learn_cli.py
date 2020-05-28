"""
Run the uncoverml pipeline for clustering, supervised learning and prediction.

.. program-output:: uncoverml --help
"""
from collections import namedtuple
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

SharedTrainingData = namedtuple('TrainingData', 
                          ['targets_all', 'x_all', 'obs_win', 'pos_win', 'field_wins', 'x_win'])

def main(config_file, partitions):
    config = ls.config.Config(config_file, learning=True)
    training_data, oos_data = _load_data(config, partitions)
    targets_all = training_data.targets_all
    x_all = training_data.x_all

    if config.cross_validate:
        crossval_results = ls.validate.local_crossval(x_all, targets_all, config)
        if crossval_results:
            ls.mpiops.run_once(crossval_results.export_crossval, config)

    _logger.info("Learning full {} model".format(config.algorithm))
    model = ls.learn.local_learn_model(x_all, targets_all, config)
    # use trained model
    if config.permutation_importance:
        ls.mpiops.run_once(
            ls.validate.permutation_importance, model, x_all, targets_all, config)

    if config.out_of_sample_validation and oos_data is not None:
        oos_targets = oos_data.targets_all
        oos_features = oos_data.x_all
        if ls.mpiops.chunk_index == 0:
            oos_results = ls.validate.out_of_sample_validation(model, oos_targets, oos_features)
            oos_results.export_scores(config)

    ls.mpiops.run_once(ls.geoio.export_model, model, config)
    if config.extents:
        ls.mpiops.run_once(_clean_temp_cropfiles, config)

    ls.geoio.deallocate_shared_training_data(training_data)

    _logger.info("Finished! Total mem = {:.1f} GB".format(_total_gb()))

def _load_data(config, partitions):
    if config.pk_load:
        if ls.mpiops.chunk_index == 0:
            x_all = pickle.load(open(config.pk_covariates, 'rb'))
            targets_all = pickle.load(open(config.pk_targets, 'rb'))
            if config.cubist or config.multicubist:
                config.algorithm_args['feature_type'] = \
                    pickle.load(open(config.pk_featurevec, 'rb'))
            _logger.warning("Using  pickled targets and covariates. Make sure you have"
                            " not changed targets file and/or covariates.")
        else:
            x_all = None
            targets_all = None
            if config.cubist or config.multicubist:
                config.algorithm_args['feature_type'] = None
        oos_data = None
    else:
        bounds = ls.geoio.get_image_bounds(config)
        if config.extents:
            if config.extents_are_pixel_coordinates:
                pw, ph = ls.geoio.get_image_pixel_res(config)
                xmin, ymin, xmax, ymax = config.extents
                xmin = xmin * pw + bounds[0][0] if xmin is not None else bounds[0][0]
                ymin = ymin * ph + bounds[1][0] if ymin is not None else bounds[1][0]
                xmax = xmax * pw + bounds[0][0] if xmax is not None else bounds[0][1]
                ymax = ymax * ph + bounds[1][0] if ymax is not None else bounds[1][1]
                target_extents = xmin, ymin, xmax, ymax
            else:
                target_extents = config.extents
            ls.geoio.crop_covariates(config)
        else:
            target_extents = bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]

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
                                        extents=target_extents)

        if config.target_search:
            if ls.mpiops.chunk_index == 0:
                # Include targets and covariates from target search
                with open(config.targetsearch_result_data, 'rb') as f:
                    ts_t = pickle.load(f)
                pos, obs, fields = ts_t.positions, ts_t.observations, ts_t.fields
            else:
                pos, obs, fields = None, None, None

            ts_t = ls.geoio.distribute_targets(pos, obs, fields)
            targets = ls.targets.merge_targets(targets, ts_t)

        # If using out-of-sample validation, split off a percentage of data before transformation
        if config.out_of_sample_validation:
            if config.oos_percentage is not None:
                num_targets = int(len(targets.observations) * config.oos_percentage)
                num_targets = len(np.array_split(np.arange(0, num_targets), 
                                  ls.mpiops.chunks)[ls.mpiops.chunk_index])
                inds = np.zeros(targets.observations.shape, dtype=bool)
                inds[:num_targets] = True
                np.random.shuffle(inds)
                oos_pos = targets.positions[inds]
                oos_obs = targets.observations[inds]
                oos_fields = {}
                for k, v in targets.fields.items():
                    oos_fields[k] = v[inds]
                oos_targets = ls.targets.Targets(oos_pos, oos_obs, oos_fields)

                _logger.info(f":mpi: oos targets = {len(oos_targets.observations)}")

                targets.positions = targets.positions[~inds]
                targets.observations = targets.observations[~inds]
                for k, v in targets.fields.items():
                    targets.fields[k] = v[~inds]

                _logger.info(f":mpi: targets = {len(targets.observations)}")

                oos_image_chunk_sets = ls.geoio.image_feature_sets(oos_targets, config)

        # Get the image chunks and their associated transforms
        image_chunk_sets = ls.geoio.image_feature_sets(targets, config)
        transform_sets = [k.transform_set for k in config.feature_sets]

        if config.raw_covariates:
            _logger.info("Saving raw data before any processing")
            ls.features.save_intersected_features_and_targets(image_chunk_sets,
                                                              transform_sets, targets, config, 
                                                              impute=False)

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
        targets_all = ls.targets.gather_targets(targets, keep, node=0)


        # Transform out-of-sample features after training data transform is performed so we use
        # the same statistics.
        if config.out_of_sample_validation:
            oos_features, keep = ls.features.transform_features(oos_image_chunk_sets, transform_sets,
                                                                config.final_transform, config)
            oos_targets = ls.targets.gather_targets(oos_targets, keep, node=0)
            oos_features = ls.features.gather_features(oos_features[keep], node=0)
            oos_data = ls.geoio.create_shared_training_data(oos_targets, oos_features)
        else:
            oos_data = None

        # Pickle data if requested.
        if ls.mpiops.chunk_index == 0:
            if config.pk_covariates and not os.path.exists(config.pk_covariates):
                pickle.dump(x_all, open(config.pk_covariates, 'wb'))
            if config.pk_targets and not os.path.exists(config.pk_targets):
                pickle.dump(targets_all, open(config.pk_targets, 'wb'))
 
    return ls.geoio.create_shared_training_data(targets_all, x_all), oos_data

def _total_gb():
    # given in KB so convert
    my_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    # total_usage = mpiops.comm.reduce(my_usage, root=0)
    total_usage = ls.mpiops.comm.allreduce(my_usage)
    return total_usage

def _clean_temp_cropfiles(config):
    shutil.rmtree(config.tmpdir)   

