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
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

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
    ls.mpiops.run_once(ls.geoio.export_model, model, config)
    # use trained model
    if config.permutation_importance:
        ls.mpiops.run_once(
            ls.validate.permutation_importance, model, x_all, targets_all, config)

    if config.out_of_sample_validation and oos_data is not None:
        oos_targets = oos_data.targets_all
        oos_features = oos_data.x_all
        oos_results = ls.validate.out_of_sample_validation(model, oos_targets, oos_features, config)
        if oos_results:
            oos_results.export_scores(config)

    if config.extents:
        ls.mpiops.run_once(_clean_temp_cropfiles, config)

    ls.geoio.deallocate_shared_training_data(training_data)
    if oos_data is not None:
        ls.geoio.deallocate_shared_training_data(oos_data)

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
        if config.out_of_sample_validation:
            _logger.warning("Can't perform out-of-sample validation when loading from pickled data")
        oos_data = None
    else:
        if not config.tabular_prediction:
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
            covariate_crs=ls.geoio.get_image_crs(config)
        else:
            target_extents = None
            covariate_crs = None

        config.n_subchunks = partitions

         # Make the targets
        _logger.info("Intersecting targets as pickled train data was not "
                     "available")
        
        targets = ls.geoio.load_targets(shapefile=config.target_file,
                                        targetfield=config.target_property,
                                        covariate_crs=covariate_crs,
                                        extents=target_extents)

        _logger.info(f":mpi:Assigned {targets.observations.shape[0]} targets")

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

        ls.targets.drop_target_values(targets, config.target_drop_values)

        # TODO: refactor out-of-sample out of script module
        # If using out-of-sample validation, split off a percentage of data before transformation
        if config.out_of_sample_validation:
            if config.oos_percentage is not None:
                num_targets = int(np.around(len(targets.observations) * config.oos_percentage))
                inds = np.zeros(targets.observations.shape, dtype=bool)
                inds[:num_targets] = True
                np.random.shuffle(inds)
                oos_pos = targets.positions[inds]
                oos_obs = targets.observations[inds]
                oos_fields = {}
                for k, v in targets.fields.items():
                    oos_fields[k] = v[inds]
                oos_targets = ls.targets.Targets(oos_pos, oos_obs, oos_fields)

                targets.positions = targets.positions[~inds]
                targets.observations = targets.observations[~inds]
                for k, v in targets.fields.items():
                    targets.fields[k] = v[~inds]

            elif config.oos_shapefile is not None:
                oos_targets = ls.geoio.load_targets(shapefile=config.oos_shapefile,
                                                    targetfield=config.oos_property,
                                                    covariate_crs=covariate_crs,
                                                    extents=target_extents)

            else:
                _logger.info("Out-of-sample validation being skipped as no 'percentage' or "
                             "'shapefile' parameter was provided.")
            if config.tabular_prediction:
                intersect_indices = ls.features.intersect_shapefile_features(
                    targets, config.feature_sets)
                oos_feature_chunks = ls.features.features_from_shapefile(
                    config.feature_sets, intersect_indices)
            else:
                oos_feature_chunks = ls.geoio.image_feature_sets(oos_targets, config)

        if config.tabular_prediction:
            intersect_indices = ls.features.intersect_shapefile_features(
                targets, config.feature_sets)
            feature_chunks = ls.features.intersect_shapefile_features(
                config.feature_sets, intersect_indices)
        else:
            feature_chunks = ls.geoio.image_feature_sets(targets, config)

        transform_sets = [k.transform_set for k in config.feature_sets]

        if config.raw_covariates:
            _logger.info("Saving raw data before any processing")
            ls.features.save_intersected_features_and_targets(feature_chunks,
                                                              transform_sets, targets, config, 
                                                              impute=False)

        if config.rank_features:
            _logger.info("Ranking features...")
            measures, features, scores = \
                ls.validate.local_rank_features(feature_chunks, transform_sets, targets, config)
            ls.mpiops.run_once(ls.geoio.export_feature_ranks, measures, features, scores, config)

        features, keep = ls.features.transform_features(feature_chunks,
                                                        transform_sets,
                                                        config.final_transform,
                                                        config)

        x_all = ls.features.gather_features(features[keep], node=0)
        targets_all = ls.targets.gather_targets(targets, keep, node=0)


        # Transform out-of-sample features after training data transform is performed so we use
        # the same statistics.
        if config.out_of_sample_validation:
            oos_features, keep = ls.features.transform_features(oos_feature_chunks, transform_sets,
                                                                config.final_transform, config)
            oos_targets = ls.targets.gather_targets(oos_targets, keep, node=0)
            oos_features = ls.features.gather_features(oos_features[keep], node=0)
            oos_data = ls.geoio.create_shared_training_data(oos_targets, oos_features)
            if ls.mpiops.chunk_index == 0 and config.oos_percentage:
                _logger.info(f"{oos_targets.observations.shape[0]} targets withheld for "
                             f"out-of-sample validation. Saved to {config.oos_targets_file}")
                oos_targets.to_geodataframe().to_file(config.oos_targets_file)
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
    my_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    total_usage = ls.mpiops.comm.allreduce(my_usage)
    return total_usage

def _clean_temp_cropfiles(config):
    shutil.rmtree(config.tmpdir)   

