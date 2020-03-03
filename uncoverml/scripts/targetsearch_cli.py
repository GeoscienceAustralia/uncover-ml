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
    if not config.extents:
        raise ValueError("Can't perform target search without specifying an extent. Provide the "
                         "'extents' block in the config file and run again.")

    config.n_subchunks = partitions
    if config.n_subchunks > 1:
        _logger.info("Memory constraint forcing {} iterations "
                     "through data".format(config.n_subchunks))
    else:
        _logger.info("Using memory aggressively: "
                             "dividing all data between nodes")

    _logger.info("Loading real targets and generating training targets...")

    # Load all 'real' targets from shapefile.
    # TODO: Drop targets from prediction area?
    real_targets = ls.geoio.load_targets(
            shapefile=config.target_file, targetfield=config.target_property, 
            covariate_crs=ls.geoio.get_image_crs(config))

    REAL_TARGETS_LABEL = 'a_real'
    # Backup original observation values by storing in `fields`
    ORIGINAL_OBSERVATIONS = 'original'
    real_targets = ls.targets.label_targets(
        real_targets, REAL_TARGETS_LABEL, backup_field=ORIGINAL_OBSERVATIONS)

    # Get random sample of points from within prediction area.
    GEN_TARGETS_LABEL = 'b_generated'
    gen_targets = ls.targets.generate_dummy_targets(
            config.extents, GEN_TARGETS_LABEL, 
            real_targets.positions.shape[0], list(real_targets.fields.keys()))

    _logger.debug(f"Class 1: {real_targets.positions.shape}, '{real_targets.observations[0]}'\t" 
                  f"Class 2: {gen_targets.positions.shape}, '{gen_targets.observations[0]}'")

    targets = ls.targets.concatenate_targets(real_targets, gen_targets)

    # Intersect targets and covariate data.
    image_chunk_sets = ls.geoio.image_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]
    features, keep = ls.features.transform_features(
            image_chunk_sets, transform_sets, config.final_transform, config)

    x_all = ls.features.gather_features(features[keep], node=0)
    targets_all = ls.targets.gather_targets(targets, keep, node=0)

    # Write out targets for debug purpses
    if ls.mpiops.chunk_index == 0:
        ls.targets.save_targets(targets_all, config.targetsearch_generated_points,
                                obs_filter=GEN_TARGETS_LABEL)
    
    # Train the model
    model = ls.models.LogisticClassifier(random_state=1)
    ls.models.apply_multiple_masked(
        model.fit, (x_all, targets_all.observations), 
        kwargs={'fields': targets_all.fields, 'lon_lat': targets_all.positions})

    # Classify
    if ls.mpiops.chunk_index == 0:
        y_star = ls.predict.predict(x_all, model, config.quantiles)
        real_ind = targets_all.observations == REAL_TARGETS_LABEL

        # Filter out generated targets
        real_pos = targets_all.positions[real_ind]
        lons, lats = real_pos.T[0], real_pos.T[1]
        likelihood = y_star[real_ind].T[2]

        # Save for debugging/visualisation
        targets_with_likelihood = np.rec.fromarrays(
            (lons, lats, likelihood), names='lon,lat,prediction_area_likelihood')
        np.savetxt(config.targetsearch_likelihood, targets_with_likelihood, 
                   fmt='%.8f,%.8f,%.8f', delimiter=',', 
                   header='lon,lat,prediction_area_likelihood')

        # Get real targets + covariate data for points where 
        #  classification threshold is met
        threshold_ind = likelihood >= config.targetsearch_threshold
        pos = real_pos[threshold_ind]
        obs = (targets_all.fields[ORIGINAL_OBSERVATIONS][real_ind])[threshold_ind]
        fields = {k: (v[real_ind])[threshold_ind] for k, v in targets_all.fields.items()}
        result_t = ls.targets.Targets(pos, obs, fields)
        result_x = (x_all[real_ind])[threshold_ind]
        # And save as binary for reuse in learn step
        with open(config.targetsearch_result_data, 'wb') as f:
            pickle.dump((result_t, result_x), f) 
        _logger.info(f"Target search complete. Found {len(pos)} targets for inclusion.")
