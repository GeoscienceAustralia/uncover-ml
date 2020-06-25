"""
This command is used to hotfix models that were trained before the
covariate transform fixes were applied. 

.. program-output:: uncoverml --help
"""
from collections import namedtuple
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

def main(config_file):
    _logger.info(
        "Running model fixer. This will inject transform information into your model. NOTE: it is "
        "crucial that your config file contains the parameters used when training the model. If "
        "this is not the case, or the training data has changed, please retrain the model using "
        "`uncoverml learn config.yaml` instead."
    )
    config = ls.config.Config(config_file, learning=True)
    # Re-run transforms on the data so we can get `feature_sets` and
    # `final_transform` objects loaded with training data statistics
    _ = _load_data(config)
    # Load the old model.
    model = _load_model(config)
    # Back it up in case we break something.
    _backup_model(config)

    if isinstance(model, tuple):
        raise TypeError(
            "Loaded model already contains extra data. You may be attempting to fix an already "
            "fixed model. Please try running your prediction with this config."
        )
    # Re-save the model. The `export_model` function now also stores
    # `feature_sets` and `final_transform` in the pickle file so they
    # can be reused in prediction.
    ls.mpiops.run_once(ls.geoio.export_model, model, config)
    if config.extents:
        ls.mpiops.run_once(_clean_temp_cropfiles, config)
    
    _logger.info(f"Finished! The model has been saved ({config.model_file}) with transform data "
        f"and this config can now be used for predictions. Note a backup of the origin model has "
        f"been made ({config.model_file + '_backup'}). It's recommended to keep this safe as this "
        f"is still experimental."
    )
    _logger.info("Total mem = {:.1f} GB".format(ls.scripts.total_gb()))

def _load_data(config):
    if config.extents:
        ls.geoio.crop_covariates(config)
    config.n_subchunks = 1

    # Make the targets
    _logger.info("Intersecting targets as pickled train data was not "
                 "available")
    if config.extents and config.extents_are_pixel_coordinates:
        pw, ph = ls.geoio.get_image_pixel_res(config)
        bounds = ls.geoio.get_image_bounds(config)
        xmin, ymin, xmax, ymax = config.extents
        xmin = xmin * pw + bounds[0][0]
        ymin = ymin * ph + bounds[1][0]
        xmax = xmax * pw + bounds[0][0]
        ymax = ymax * ph + bounds[1][0]
        target_extents = xmin, ymin, xmax, ymax
    else:
        target_extents = config.extents

    targets = ls.geoio.load_targets(shapefile=config.target_file,
                                    targetfield=config.target_property,
                                    covariate_crs=ls.geoio.get_image_crs(config),
                                    extents=target_extents)
                                        
    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.geoio.image_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    features, keep = ls.features.transform_features(image_chunk_sets,
                                                    transform_sets,
                                                    config.final_transform,
                                                    config)

    # TODO: Update this once target search is fixed.
    if config.target_search:
        _logger.warning("Target search transformations are still not fixed. If your model was "
            "trained on data that includes data collected from `targetsearch`, then the "
            "transformation statistics for your data will only be based on the data loaded from "
            "TIF files. While your model will still work, this may have negative impacts on "
            "the model's performance."
        )

def _clean_temp_cropfiles(config):
    shutil.rmtree(config.tmpdir)   

def _load_model(config):
    with open(config.model_file, 'rb') as f:
        return pickle.load(f)

def _backup_model(config):
    shutil.copyfile(config.model_file, config.model_file + '_backup')
