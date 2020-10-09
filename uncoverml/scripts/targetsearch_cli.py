"""
Run the uncoverml pipeline for clustering, supervised learning and prediction.

.. program-output:: uncoverml --help
"""
import logging
import pickle
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
    config = ls.config.Config(config_file, learning=True, predicting=True)
    config.n_subchunks = partitions
    bounds = ls.geoio.get_image_bounds(config)
    if config.targetsearch_extents is None:
        _logger.info("No target search extents were provided, looking for targets in full "
                     "feature space.")
        config.targetsearch_extents = bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]
        config.tse_are_pixel_coordinates = False
    else:
        if config.tse_are_pixel_coordinates:
            pw, ph = ls.geoio.get_image_pixel_res(config)
            xmin, ymin, xmax, ymax = config.extents
            xmin = xmin * pw + bounds[0][0] if xmin is not None else bounds[0][0]
            ymin = ymin * ph + bounds[1][0] if ymin is not None else bounds[1][0]
            xmax = xmax * pw + bounds[0][0] if xmax is not None else bounds[0][1]
            ymax = ymax * ph + bounds[1][0] if ymax is not None else bounds[1][1]
            config.targetsearch_extents = xmin, ymin, xmax, ymax

    _logger.info("Loading real targets and generating training targets...")

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

    # Load all 'real' targets from shapefile.
    # TODO: Drop targets from prediction area?
    real_targets = ls.geoio.load_targets(
            shapefile=config.target_file, targetfield=config.target_property, 
            covariate_crs=ls.geoio.get_image_crs(config), extents=target_extents)
    num_targets = ls.mpiops.count_targets(real_targets)

    REAL_TARGETS_LABEL = 'a_real'
    # Backup original observation values by storing in `fields`
    ORIGINAL_OBSERVATIONS = 'original'
    real_targets = ls.targets.label_targets(
        real_targets, REAL_TARGETS_LABEL, backup_field=ORIGINAL_OBSERVATIONS)

    # Get random sample of points from within prediction area.
    GEN_TARGETS_LABEL = 'b_generated'
    gen_targets = ls.targets.generate_dummy_targets(
            config.targetsearch_extents, GEN_TARGETS_LABEL, 
            num_targets, field_keys=list(real_targets.fields.keys()))

    _logger.debug(f"Class 1: {real_targets.positions.shape}, '{real_targets.observations[0]}'\t" 
                  f"Class 2: {gen_targets.positions.shape}, '{gen_targets.observations[0]}'")

    targets = ls.targets.merge_targets(real_targets, gen_targets)

    # Intersect targets and covariate data.
    image_chunk_sets = ls.geoio.image_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]
    features, keep = ls.features.transform_features(
            image_chunk_sets, transform_sets, config.final_transform, config)

    x_all = ls.features.gather_features(features[keep], node=0)
    targets_all = ls.targets.gather_targets(targets, keep, node=0)

    if ls.mpiops.leader_world:
        # Write out targets for debug purpses
        ls.targets.save_targets(targets_all, config.targetsearch_generated_points,
                                obs_filter=GEN_TARGETS_LABEL)

        # Train the model and classify
        model = ls.models.LogisticClassifier(random_state=1)
        ls.models.apply_multiple_masked(model.fit, (x_all, targets_all.observations))
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
        # result_x = (x_all[real_ind])[threshold_ind]
        # And save as binary for reuse in learn step
        with open(config.targetsearch_result_data, 'wb') as f:
            pickle.dump(result_t, f) 
        _logger.info(f"Target search complete. Found {len(pos)} targets for inclusion.")
