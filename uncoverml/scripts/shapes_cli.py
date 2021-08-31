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
    log.info(f"Adding group col {config.output_group_col_name} in input shaefile {input_shapefile}")
    gdf = gpd.read_file(input_shapefile)

    rows = config.spatial_grouping_args['rows']
    cols = config.spatial_grouping_args['cols']

    polygons = ls.resampling.create_grouping_polygons_from_geo_df(rows, cols, gdf)

    df_to_concat = []

    for i, p in enumerate(polygons):
        df = gdf[gdf[ls.resampling.GEOMETRY].within(p)]
        df = df.copy()
        if df.shape[0]:
            df[config.output_group_col_name] = i
            df_to_concat.append(df)
        else:
            log.debug('{}th {} does not contain any sample'.format(i, p))
    output_gdf = gpd.pd.concat(df_to_concat)
    output_gdf.to_file(output_shapefile)
    log.info(f"Wrote {output_shapefile} file with spatial groups added in column {config.output_group_col_name}")


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
    log.info(f"Spliting shapefile {input_shapefile} in train and oss shapefiles")
    gdf = gpd.read_file(input_shapefile)
    all_groups = np.unique(gdf[config.split_group_col_name])
    train_groups, oos_groups = train_test_split(all_groups, test_size=config.split_oos_fraction)
    train_gdf = gdf[gdf[config.split_group_col_name].isin(train_groups)]
    oos_gdf = gdf[gdf[config.split_group_col_name].isin(oos_groups)]
    train_gdf.to_file(config.train_shapefile)
    oos_gdf.to_file(config.oos_shapefile)
    oos_gdf.to_file(config.oos_shapefile)
    log.info(f"saved train shapefile {config.train_shapefile}")
    log.info(f"saved oss shapefile {config.oos_shapefile}")
