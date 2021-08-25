import os
import logging
import click
import uncoverml as ls
import uncoverml.mllog
import uncoverml.resampling
import uncoverml.config

log = logging.getLogger(__name__)


@click.command()
@click.argument('pipeline_file')
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def cli(pipeline_file, verbosity):
    ls.mllog.configure(verbosity)
    config = ls.config.Config(pipeline_file)

    config.resample = True

    transforms = []

    if config.value_resampling_args:
        transforms.append((ls.resampling.resample_by_magnitude, config.value_resampling_args))
    if config.spatial_resampling_args:
        transforms.append((ls.resampling.resample_spatially, config.spatial_resampling_args))
    input_shapefile = config.target_file
    output_shapefile = config.resampled_output
    print(output_shapefile)
    target_field = config.target_property

    gdf = None

    for i, (func, kwargs) in enumerate(transforms):
        gdf = func(input_shapefile, target_field, **kwargs)

    gdf.to_file(output_shapefile)

    log.info("Resampling complete. Resampled targets saved to '{}'".format(output_shapefile))
