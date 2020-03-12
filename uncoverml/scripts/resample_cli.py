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

resampling_techniques = {'value': ls.resampling.resample_by_magnitude,
                         'space': ls.resampling.resample_spatially}


def main(config_file, validation_file=None, validation_points=100):
    config = ls.config.Config(config_file, resampling=True)

    filename = os.path.splitext(os.path.basename(config.target_file))[0]
    out_shpfile = config.resampled_shapefile_dir.format(filename)

    final_shpfile = abspath(out_shpfile)
    _transforms = []
    if config.value_resampling_args:
        _transforms.append(('value', config.value_resampling_args))
    if config.spatial_resampling_args:
        _transforms.append(('space', config.spatial_resampling_args))

    for i, r in enumerate(_transforms):
        _shpfile = final_shpfile if i == len(_transforms) - 1 \
            else tempfile.mktemp(suffix='.shp', dir='/tmp')

        input_shpfile = config.target_file if i == 0 else out_shpfile

        # just create the validation shape during last sampling step
        validation = validation_file \
           if i == len(_transforms) - 1 else None

        out_shpfile = resampling_techniques[r[0]](
            input_shpfile, _shpfile,
            target_field=config.target_property,
            validation_file=validation,
            validation_points=validation_points,
            **r[1]
        )

        _logger.info('Output shapefile is located in {}'.format(out_shpfile))
