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
    config = ls.config.Config(config_file, predicting=True, shiftmap=True)
    if config.pk_load:
        raise ValueError("Can't create covariate shiftmap when loading from pickled data. Remove "
                         "'pickling' block from config and provide 'targets' and 'features' "
                         "blocks.")

    # Force algortihm - this is purely for debug log messages
    config.algorithm = 'logistic'
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

    real_targets = ls.geoio.load_targets(shapefile=config.target_file,
                                        targetfield=config.target_property,
                                        covariate_crs=ls.geoio.get_image_crs(config),
                                        extents=target_extents)
    
    ls.mpiops.comm_world.barrier()

    # User can provide their own 'query' targets for training shapemap, or we can
    # generate points.
    if config.shiftmap_targets:
        dummy_targets = ls.geoio.load_targets(shapefile=config.shiftmap_targets,
                                              covariate_crs=ls.geoio.get_image_crs(config),
                                              extents=config.extents)
        dummy_targets = ls.targets.label_targets(dummy_targets, 'query')
        real_targets = ls.targets.label_targets(real_targets, 'training')
        targets = ls.targets.merge_targets(dummy_targets, real_targets)
    else:
        bounds = ls.geoio.get_image_bounds(config)
        bounds = (bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1])
        num_targets = ls.mpiops.count_targets(real_targets)
        targets = ls.targets.generate_covariate_shift_targets(real_targets, bounds, num_targets) 
                                                              

    image_chunk_sets = ls.geoio.image_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]
    features, keep = ls.features.transform_features(image_chunk_sets,
                                                    transform_sets,
                                                    config.final_transform,
                                                    config)
    x_all = ls.features.gather_features(features[keep], node=0)
    targets_all = ls.targets.gather_targets(targets, keep, node=0)
    if ls.mpiops.leader_world:
        ls.targets.save_targets(targets_all, config.shiftmap_points, 'query')
        model = ls.models.LogisticClassifier(random_state=1)
        ls.models.apply_multiple_masked(model.fit, (x_all, targets_all.observations),
                                        kwargs={'fields': targets_all.fields,
                                                'lon_lat': targets_all.positions})
    else:
        model = None

    model = ls.mpiops.comm_world.bcast(model, root=0)

    # The below is essentially duplicating the 'predict' command
    # should refactor to reuse it
    image_shape, image_bbox, image_crs = ls.geoio.get_image_spec(model, config)

    predict_tags = model.get_predict_tags()
    config.outbands = len(predict_tags)

    image_out = ls.geoio.ImageWriter(image_shape, image_bbox, image_crs,
                                     config.n_subchunks, config.shiftmap_file, config.outbands,
                                     band_tags=predict_tags,
                                     **config.geotif_options)

    for i in range(config.n_subchunks):
        _logger.info("starting to render partition {}".format(i+1))
        ls.predict.render_partition(model, i, image_out, config)

    image_out.close()

    if config.thumbnails:
        image_out.output_thumbnails(config.thumbnails)

    if config.extents:
        ls.mpiops.run_once(_clean_temp_cropfiles, config)


def _clean_temp_cropfiles(config):
    shutil.rmtree(config.tmpdir)   

