import logging
import click
import numpy as np
from sklearn.model_selection import train_test_split
import geopandas as gpd
import uncoverml as ls
import uncoverml.mllog
import uncoverml.resampling
import uncoverml.config

log = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def cli(verbosity):
    ls.mllog.configure(verbosity)


@cli.command()
@click.argument('pipeline_file')
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def group(pipeline_file, verbosity):
    """add spatial groups to the shapefile using """

    ls.mllog.configure(verbosity)
    config = ls.config.Config(pipeline_file)

    input_shapefile = config.target_file
    output_shapefile = config.grouped_output
    gdf = gpd.read_file(input_shapefile)

    rows = config.spatial_grouping_args['rows']
    cols = config.spatial_grouping_args['cols']

    polygons = ls.resampling.create_grouping_polygons_from_geo_df(rows, cols, gdf)

    df_to_concat = []

    for i, p in enumerate(polygons):
        df = gdf[gdf[ls.resampling.GEOMETRY].within(p)]
        df = df.copy()
        if df.shape[0]:
            df['group_col'] = i
            df_to_concat.append(df)
        else:
            log.debug('{}th {} does not contain any sample'.format(i, p))
    output_gdf = gpd.pd.concat(df_to_concat)
    output_gdf.to_file(output_shapefile)


@cli.command()
@click.argument('pipeline_file')
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def split(pipeline_file, verbosity):
    """split a shapefile into two - possibly to use for training and complete oss verification"""
    ls.mllog.configure(verbosity)
    config = ls.config.Config(pipeline_file)
    input_shapefile = config.grouped_output
    gdf = gpd.read_file(input_shapefile)
    all_groups = np.unique(gdf[config.split_group_col_name])
    train_groups, oos_groups = train_test_split(all_groups, test_size=config.split_oos_fraction)
    train_gdf = gdf[gdf[config.split_group_col_name].isin(train_groups)]
    oos_gdf = gdf[gdf[config.split_group_col_name].isin(oos_groups)]
    train_gdf.to_file(config.train_shapefile)
    oos_gdf.to_file(config.oos_shapefile)
    oos_gdf.to_file(config.oos_shapefile)
