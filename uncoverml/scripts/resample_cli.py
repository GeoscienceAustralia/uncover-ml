import tempfile
import os
from os.path import abspath, exists, splitext
from os import remove
import logging

import click

import uncoverml as ls
import uncoverml.mllog
import uncoverml.config
import uncoverml.resampling


_logger = logging.getLogger(__name__)


def main(config_file, validation=False, validation_points=100):
    if validation is True:
        raise NotImplementedError("Validation for resampling needs to "
            "be reimplmented.")

    config = ls.config.Config(config_file, resampling=True)

    filename = os.path.splitext(os.path.basename(config.target_file))[0]
    out_shpfile = config.resampled_shapefile_dir.format(filename)
    transforms = []

    if config.value_resampling_args:
        transforms.append((ls.resampling.resample_by_magnitude, config.value_resampling_args))
    if config.spatial_resampling_args:
        transforms.append((ls.resampling.resample_spatially, config.spatial_resampling_args))

    input_shapefile = config.target_file
    output_shapefile = config.resampled_shapefile_dir.format(
        os.path.splitext(os.path.basename(config.target_file))[0])
    target_field = config.target_property
    gdf = None

    for i, (func, kwargs) in enumerate(transforms):
        input_data = input_shapefile if gdf is None else gdf
        gdf = func(input_data, target_field, **kwargs)

    gdf.to_file(output_shapefile)

    _logger.info("Resampling complete. Resampled targets saved to '{}'".format(output_shapefile))
