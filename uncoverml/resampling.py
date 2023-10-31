"""
Module for shapefile resampling methods.
This code was originailly developed by Sudipta Basak.
(https://github.com/basaks)

See `uncoverml.scripts.shiftmap_cli` for a resampling CLI.
"""
import logging
from typing import List, Optional
import geopandas as gpd
import pandas as pd
import pandas.core.algorithms as algos
import numpy as np
from shapely.geometry import Polygon
from fiona.errors import DriverError

BIN = 'bin'
GEOMETRY = 'geometry'
log = logging.getLogger(__name__)


def bootstrap_data_indicies(population, samples=None, random_state=1):
    samples = population if samples is None else samples
    return np.random.RandomState(random_state).randint(0, population, samples)


def prepapre_dataframe(data, target_field: str, fields_to_keep: Optional[List[str]] = None):
    try:
        gdf = gpd.read_file(data)
    except DriverError:
        log.error(
            "Couldn't read data for resampling. Ensure a valid "
            "shapefile path or Targets object is being provided "
            "as input."
        )
        raise
    if fields_to_keep is None:
        fields_to_keep = gdf.columns  # target will be in this
    else:
        fields_to_keep += [target_field]

    return filter_fields(fields_to_keep, gdf)


def filter_fields(fields_to_keep, gdf):
    fields_to_keep = [GEOMETRY] + list(fields_to_keep)  # add geometry
    fields_to_keep = list(set(fields_to_keep))
    original_fields = gdf.columns
    for f in fields_to_keep:
        if f not in original_fields:
            raise RuntimeError("field '{}' must exist in shapefile".format(f))
    gdf_out = gdf[fields_to_keep]
    return gdf_out


def resample_by_magnitude(input_data, target_field, bins=10, interval='linear', undersample=False,
                          fields_to_keep: Optional[List[str]] = None, bootstrap=True, output_samples=None,
                          validation=False, validation_points=100):
    """
    Parameters
    ----------
    input_gdf : geopandas.GeoDataFrame
        Geopandas dataframe containing targets to be resampled.
    target_field : str
        target field name based on which resampling is performed. Field
        must exist in the input_shapefile
    bins : int
        number of bins for sampling
    fields_to_keep : list
        of strings to store in the output shapefile
    bootstrap : bool, optional
        whether to sample with replacement or not
    output_samples : int, optional
        number of samples in the output shpfile. If not provided, the 
        output samples will be assumed to be the same as the original 
        shapefile
    validation : bool, optional
        validation file name
    validation_points : int, optional
        approximate number of points in the validation shapefile
    :param undersample:
    :param interval:

    """
    log.info("Resampling shapefile by magnitude of target values")
    if bootstrap and validation:
        raise ValueError('bootstrapping should not be use while'
                         'creating a validation shapefile.')
    if interval not in ['percentile', 'linear']:
        log.warning("Interval method '{}' not recognised, defaulting to 'percentile'"
                    .format(interval))
        interval = 'linear'
    gdf_out = prepapre_dataframe(input_data, target_field, fields_to_keep)

    if validation:
        assert validation_points is not None
        assert validation_points != 0
        assert validation_points >= bins
        assert validation_points + output_samples <= gdf_out.shape[0]

        # the idea is stolen from pandas.qcut
    # pd.qcut does not work for cases when it result in non-unique bin edges
    target = gdf_out[target_field].values
    if interval == 'percentile':
        bin_edges = np.quantile(np.asarray(target), np.linspace(0, 1, bins + 1))
    elif interval == 'linear':
        bin_edges = np.linspace(np.min(target), np.max(target), bins + 1)

    log.info(f"resampling using {interval} of target values")

    result = pd.core.reshape.tile._bins_to_cuts(target, bin_edges,
                                                labels=False,
                                                duplicates='drop',
                                                include_lowest=True)

    # add to output df for sampling
    gdf_out[BIN] = result[0]

    dfs_to_concat = []
    validation_dfs_to_concat = []
    approximate_samples_per_bin = gdf_out.shape[0] // bins
    output_samples_per_bin = output_samples // bins
    validation_samples_per_bin = validation_points // bins
    assert approximate_samples_per_bin >= validation_samples_per_bin + output_samples_per_bin
    gb = gdf_out.groupby(BIN)
    for i, (b, gr) in enumerate(gb):
        if bootstrap:
            if undersample:
                dfs_to_concat.append(gr.sample(n=approximate_samples_per_bin, replace=bootstrap))
            else:
                if gr.shape[0] < approximate_samples_per_bin:
                    dfs_to_concat.append(gr.sample(n=approximate_samples_per_bin, replace=bootstrap))
                else:
                    dfs_to_concat.append(gr)
        else:
            _df, v_df = _sample_without_replacement(
                gr, output_samples_per_bin,
                gr.shape[0] - output_samples_per_bin
            )
            assert gr.shape[0] - output_samples_per_bin + output_samples_per_bin <= gr.shape[0]
            dfs_to_concat.append(_df)
            validation_dfs_to_concat.append(v_df)

    final_df = pd.concat(dfs_to_concat)
    final_df.sort_index(inplace=True)
    output_gdf = final_df.drop(BIN, axis=1)
    if validation:
        validation_df = pd.concat(validation_dfs_to_concat)
        return output_gdf, validation_df
    else:
        return output_gdf, gpd.GeoDataFrame()


def resample_spatially(input_shapefile, target_field, rows=10, cols=10,
                       undersample=False,
                       fields_to_keep: Optional[List[str]] = None, bootstrap=True, output_samples=Optional[int],
                       validation = False,
                       validation_points=100):
    """
    Parameters
    ----------
    input_shapefile
    output_shapefile
    target_field: str
        target field name based on which resampling is performed. Field 
        must exist in the input_shapefile
    rows: int, optional
        number of bins in y
    cols: int, optional
        number of bins in x
    fields_to_keep: list of strings to store in the output shapefile
    bootstrap: bool, optional
        whether to sample with replacement or not
    output_samples: int, optional
        number of samples in the output shpfile. If not provided, the 
        output samples will be assumed to be the same as the original 
        shapefile
    validation_points: int, optional
        approximate number of points in the validation shapefile

    Returns
    -------
    output_shapefile name

    """
    log.info("Resampling shapefile by using spatial binning")
    if bootstrap and validation:
        raise ValueError('bootstrapping should not be use while'
                         'creating a validation shapefile.')

    gdf_out = prepapre_dataframe(input_shapefile, target_field, fields_to_keep)

    polygons = create_grouping_polygons_from_geo_df(rows, cols, gdf_out)

    df_to_concat = []
    validation_dfs_to_concat = []

    total_samples = output_samples if output_samples else gdf_out.shape[0]
    samples_per_group = total_samples // len(polygons)

    validate_array = np.ones(len(polygons), dtype=np.bool)
    if len(polygons) > validation_points:
        validate_array[validation_points:] = False
        np.random.shuffle(validate_array)

    for i, p in enumerate(polygons):
        df = gdf_out[gdf_out[GEOMETRY].within(p)]
        if df.shape[0]:
            if bootstrap:
                # should probably discard if df.shape[0] < 10% of
                # samples_per_group
                if undersample:
                    df_to_concat.append(df.sample(n=samples_per_group, replace=bootstrap))
                else:
                    if df.shape[0] < samples_per_group:
                        df_to_concat.append(df.sample(n=samples_per_group, replace=bootstrap))
                    else:
                        df_to_concat.append(df)
            else:
                _df, v_df = _sample_without_replacement(df, samples_per_group, validate_array[i])
                df_to_concat.append(_df)
                validation_dfs_to_concat.append(v_df)
        else:
            log.debug('{}th {} does not contain any sample'.format(i, p))
    output_gdf = pd.concat(df_to_concat)
    if validation:
        validation_df = pd.concat(validation_dfs_to_concat)
        return output_gdf, validation_df
    else:
        return output_gdf, pd.DataFrame()


def create_grouping_polygons_from_geo_df(rows, cols, gdf_out):
    minx, miny, maxx, maxy = gdf_out[GEOMETRY].total_bounds
    x_grid = np.linspace(minx, maxx, num=cols + 1)
    y_grid = np.linspace(miny, maxy, num=rows + 1)
    polygons = []
    for xs, xe in zip(x_grid[:-1], x_grid[1:]):
        for ys, ye in zip(y_grid[:-1], y_grid[1:]):
            polygons.append(Polygon([(xs, ys), (xs, ye), (xe, ye), (xe, ys)]))
    return polygons


def _sample_without_replacement(df: pd.DataFrame, training_points, validation_points):
    """
    Parameters
    ----------
    df
    training_points
    validation_points: bool
        whether to create a validate df
        if False, second dataframe returned will be empty
    Returns
    -------

    """

    if df.shape[0] >= training_points + validation_points:
        # if enough points take the number of samples
        # the second df returned makes up the validation shapefile
        # _df = df.sample(n=samples_per_group + 1, replace=False)
        df = df.sample(frac=1)
        return df[:training_points], df[training_points:]
    else:
        # else take everything, this will lead to uncertain number of
        # points in the resulting shapefile
        # return an empty df for the validation set for this bin
        return df, gpd.GeoDataFrame(columns=df.columns)
