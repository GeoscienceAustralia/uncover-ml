import tempfile
from os.path import abspath
import geopandas as gpd
import pandas as pd
import pandas.core.algorithms as algos
import numpy as np
from shapely.geometry import Polygon
import logging
from uncoverml.config import ConfigException


BIN = 'bin'
GEOMETRY = 'geometry'
log = logging.getLogger(__name__)


def filter_fields(fields_to_keep, input_shapefile):
    gdf = gpd.read_file(input_shapefile)
    fields_to_keep = [GEOMETRY] + list(fields_to_keep)  # add geometry
    original_fields = gdf.columns
    for f in fields_to_keep:
        if f not in original_fields:
            raise RuntimeError("field '{}' must exist in shapefile".format(f))
    gdf_out = gdf[fields_to_keep]
    return gdf_out


def strip_shapefile(input_shapefile, output_shapefile, *fields_to_keep):
    """
    Parameters
    ----------
    input_shapefile
    output_shapefile
    args features to keep in the output shapefile
    Returns
    -------
    """

    gdf_out = filter_fields(fields_to_keep, input_shapefile)
    gdf_out.to_file(output_shapefile)


def resample_by_magnitude(input_shapefile, output_shapefile,
                          target_field, bins=10,
                          *fields_to_keep, bootstrap=True,
                          output_samples=None):
    """
    Parameters
    ----------
    input_shapefile: str
    output_shapefile: str
    target_field: str
        target field for sampling
    bins: int
        number of bins for sampling
    fields_to_keep: list
        of strings to store in the output shapefile
    bootstrap: bool
        whether to sample with replacement or not
    output_samples: number of samples in the output shapefile
    Returns
    -------

    """
    log.info("resampling shapefile by values")

    if len(fields_to_keep):
        fields_to_keep.append(target_field)
    else:
        fields_to_keep = [target_field]
    gdf_out = filter_fields(fields_to_keep, input_shapefile)

    # the idea is stolen from pandas.qcut
    # pd.qcut does not work for cases when it result in non-unique bin edges
    target = gdf_out[target_field].values
    bin_edges = algos.quantile(
        np.unique(target), np.linspace(0, 1, bins+1))
    result = pd.tools.tile._bins_to_cuts(target, bin_edges,
                                         labels=False,
                                         include_lowest=True)
    # add to output df for sampling
    gdf_out[BIN] = result

    dfs_to_concat = []
    total_samples = output_samples if output_samples else gdf_out.shape[0]
    samples_per_bin = total_samples // bins

    gb = gdf_out.groupby(BIN)
    for b, gr in gb:
        if bootstrap:
            dfs_to_concat.append(gr.sample(n=samples_per_bin,
                                           replace=bootstrap))
        else:
            if gr.shape[0] > samples_per_bin:
                dfs_to_concat.append(gr.sample(n=samples_per_bin,
                                               replace=bootstrap))
            elif gr.shape[0] <= samples_per_bin:
                dfs_to_concat.append(gr)

    final_df = pd.concat(dfs_to_concat)
    final_df.sort_index(inplace=True)
    final_df.drop(BIN, axis=1).to_file(output_shapefile)

    return output_shapefile


def resample_spatially(input_shapefile,
                       output_shapefile,
                       target_field,
                       rows=10,
                       cols=10,
                       *fields_to_keep,
                       bootstrap=True,
                       output_samples=None):
    """
    Parameters
    ----------
    input_shapefile
    output_shapefile
    rows: int
        number of bins in y
    cols: int
        number of bins in x
    fields_to_keep: list of strings to store in the output shapefile
    bootstrap: bool
        whether to sample with replacement or not
    output_samples: number of samples in the output shpfile
    Returns
    -------

    """
    log.info("resampling shapefile spatially")

    if len(fields_to_keep):
        fields_to_keep.append(target_field)
    else:
        fields_to_keep = [target_field]

    gdf_out = filter_fields(fields_to_keep, input_shapefile)

    minx, miny, maxx, maxy = gdf_out[GEOMETRY].total_bounds
    x_grid = np.linspace(minx, maxx, num=cols+1)
    y_grid = np.linspace(miny, maxy, num=rows+1)

    polygons = []
    for xs, xe in zip(x_grid[:-1], x_grid[1:]):
        for ys, ye in zip(y_grid[:-1], y_grid[1:]):
            polygons.append(Polygon([(xs, ys), (xs, ye), (xe, ye), (xe, ys)]))

    df_to_concat = []

    total_samples = output_samples if output_samples else gdf_out.shape[0]
    samples_per_group = total_samples // len(polygons)

    for i, p in enumerate(polygons):
        df = gdf_out[gdf_out[GEOMETRY].within(p)]
        # should probably discard if df.shape[0] < 10% of samples_per_group
        if df.shape[0]:
            if bootstrap:
                df_to_concat.append(df.sample(n=samples_per_group,
                                              replace=bootstrap))
            else:
                if df.shape[0] > samples_per_group:
                    df_to_concat.append(df.sample(n=samples_per_group,
                                                  replace=bootstrap))
                elif df.shape[0] <= samples_per_group:
                    df_to_concat.append(df)
        else:
            log.info('{}th {} does not contain any sample'.format(i, p))
    final_df = pd.concat(df_to_concat)
    final_df.to_file(output_shapefile)
    return output_shapefile


resampling_techniques = {'value': resample_by_magnitude,
                         'spatial': resample_spatially}


def resample_shapefile(config, outfile=None):
    shapefile = config.target_file

    if not config.resample:
        return shapefile
    else:  # sample shapefile
        if not outfile:
            final_shpfile = tempfile.mktemp(suffix='.shp',
                                            dir=config.output_dir)
        else:
            final_shpfile = abspath(outfile)

        number_of_transforms = len(config.resample)

        for i, r in enumerate(config.resample):
            for k in r:
                if k not in resampling_techniques.keys():
                    raise ConfigException("Resampling must be 'value' or "
                                          "'spatial'")

                int_shpfile = final_shpfile if i == number_of_transforms -1 \
                    else tempfile.mktemp(suffix='.shp', dir=config.output_dir)

                input_shpfile = shapefile if i == 0 else out_shpfile

                out_shpfile = resampling_techniques[k](
                    input_shpfile, int_shpfile,
                    target_field=config.target_property,
                    ** r[k]['arguments']
                    )
        log.info('Output shapefile is {}'.format(out_shpfile))
        return out_shpfile