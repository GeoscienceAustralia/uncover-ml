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
import uncoverml.scripts
from uncoverml.transforms import StandardiseTransform


_logger = logging.getLogger(__name__)
# warnings.showwarning = warn_with_traceback
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def main(config_file, subsample_fraction):
    config = ls.config.Config(config_file, clustering=True)

    for f in config.feature_sets:
        if not f.transform_set.global_transforms:
            raise ValueError("Standardise transform must be used for kmeans")
        for t in f.transform_set.global_transforms:
            if not isinstance(t, StandardiseTransform):
                raise ValueError("Only standardise transform is allowed for kmeans")

    config.subsample_fraction = subsample_fraction
    if config.subsample_fraction < 1:
        _logger.info("Memory contstraint: using {:2.2f}%"
                     " of pixels".format(config.subsample_fraction * 100))
    else:
        _logger.info("Using memory aggressively: dividing all data between nodes")

    if config.semi_supervised:
        semisupervised(config)
    else:
        unsupervised(config)
    _logger.info("Finished! Total mem = {:.1f} GB".format(ls.scripts.total_gb()))

def semisupervised(config):

    # make sure we're clear that we're clustering
    config.cubist = False
    # Get the taregts
    targets = ls.geoio.load_targets(shapefile=config.class_file,
                                    targetfield=config.class_property)

    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.geoio.semisupervised_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    features, _ = ls.features.transform_features(image_chunk_sets,
                                                 transform_sets,
                                                 config.final_transform,
                                                 config)
    features, classes = ls.features.remove_missing(features, targets)
    indices = np.arange(classes.shape[0], dtype=int)

    config.n_classes = ls.cluster.compute_n_classes(classes, config)
    model = ls.cluster.KMeans(config.n_classes, config.oversample_factor)
    _logger.info("Clustering image")
    model.learn(features, indices, classes)
    ls.mpiops.run_once(ls.geoio.export_model, model, config)

def unsupervised(config):
    # make sure we're clear that we're clustering
    config.cubist = False
    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.geoio.unsupervised_feature_sets(config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    features, _ = ls.features.transform_features(image_chunk_sets,
                                                 transform_sets,
                                                 config.final_transform,
                                                 config)

    features, _ = ls.features.remove_missing(features)
    model = ls.cluster.KMeans(config.n_classes, config.oversample_factor)
    _logger.info("Clustering image")
    model.learn(features)
    ls.mpiops.run_once(ls.geoio.export_model, model, config)