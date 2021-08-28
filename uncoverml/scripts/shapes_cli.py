import os
import logging
import click
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
    fields_to_keep = config.grouping_fields_to_keep
    gdf = gpd.read_file(input_shapefile)
    if not len(fields_to_keep):  # if empty list or no list supplied
        fields_to_keep = gdf.columns
    gdf_out = ls.resampling.filter_fields(fields_to_keep, gdf)

    rows = config.spatial_grouping_args['rows']
    cols = config.spatial_grouping_args['cols']

    polygons = ls.resampling.create_grouping_polygons_from_geo_df(rows, cols, gdf_out)

    df_to_concat = []

    # aem_data = aem_data.groupby(cluster_line_no).apply(utils.add_delta, conf=conf)
    for i, p in enumerate(polygons):
        df = gdf_out[gdf_out[ls.resampling.GEOMETRY].within(p)]
        if df.shape[0]:
            df['group_col'] = i
            df_to_concat.append(df)
        else:
            log.debug('{}th {} does not contain any sample'.format(i, p))
    output_gdf = gpd.pd.concat(df_to_concat)
    output_gdf.to_file(output_shapefile)
    import IPython; IPython.embed(); import sys; sys.exit()


@click.command()
@click.argument('pipeline_file')
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def split(pipeline_file, verbosity):
    """split a shapefile into two - possibly to use for training and complete oss verification"""
    pass
