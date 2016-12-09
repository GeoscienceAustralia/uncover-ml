import logging
import click
import glob
import numpy as np
import warnings
from numpy import ma
from osgeo import gdal
from os.path import join, abspath, basename, isdir
import csv
from uncoverml import mpiops
from uncoverml import geoio
from uncoverml import features
from uncoverml.mllog import warn_with_traceback

log = logging.getLogger(__name__)
warnings.showwarning = warn_with_traceback


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO)


def _join_dicts(dicts):
    if dicts is None:
        return
    d = {k: v for D in dicts for k, v in D.items()}
    return d


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('report_file')
@click.option('-p', '--partitions', type=int, default=100,
              help='how many partitions to read each file in? '
                   'Higher partitions require less memory.')
@click.option('-e', '--extension',
              type=str, default='tif',
              help='geotif extension, e.g., `tif`, or `tiff`')
def inspect(input_dir, report_file, partitions, extension):
    input_dir = abspath(input_dir)
    if isdir(input_dir):
        log.info('Reading tifs from {}'.format(input_dir))
        tifs = glob.glob(join(input_dir, '*.' + extension))
    else:
        log.info('Reporting geoinfo for {}'.format(input_dir))
        tifs = [input_dir]

    with open(report_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(['FineName', 'band', 'NoDataValue', 'rows', 'cols',
                         'Min', 'Max', 'Mean', 'Std',
                         'DataType', 'Categories', 'NanCount'])
        process_tifs = np.array_split(tifs, mpiops.chunks)[mpiops.chunk_index]

        stats = []  # process geotiff stats including multibanded geotif
        for t in process_tifs:
            stats.append(get_stats(t, partitions))

        # gather all process geotif stats in stats dict
        stats = _join_dicts(stats)

        # global gather in root
        stats = _join_dicts(mpiops.comm.gather(stats, root=0))

        if mpiops.chunk_index == 0:
            for k, v in stats.items():
                write_rows(v, writer)


def write_rows(stats, writer):
    writer.writerow(stats)


def get_stats(tif, partitions=100):
    ds = gdal.Open(tif, gdal.GA_ReadOnly)
    number_of_bands = ds.RasterCount

    if ds.RasterCount > 1:
        d = {}
        log.info('Found multibanded geotif {}'.format(basename(tif)))
        for b in range(number_of_bands):
            d['{tif}_{b}'.format(tif=tif, b=b)] = band_stats(ds, tif, b + 1,
                                                             partitions)
        return d
    else:
        log.info('Found single band geotif {}'.format(basename(tif)))
        return {tif: band_stats(ds, tif, 1, partitions)}


def band_stats(ds, tif, band_no, partitions=100):
    log.info('Calculating band stats for {} band {} in {} partitions'.format(
        tif, band_no, partitions))
    band = ds.GetRasterBand(band_no)
    # For statistics calculation
    stats = band.ComputeStatistics(False)
    no_data_val = band.GetNoDataValue()
    data_type = get_datatype(band)
    image_source = geoio.RasterioImageSource(tif)

    if data_type is 'Categorical':
        no_categories = stats[1] - stats[0] + 1
    else:
        no_categories = np.nan

    l = [basename(tif), band_no, no_data_val,
         ds.RasterYSize, ds.RasterXSize,
         stats[0], stats[1],
         stats[2], stats[3],
         data_type, no_categories,
         image_nans(image_source, partitions)]
    ds = None  # close dataset
    return [str(a) for a in l]


def get_datatype(band):
    data_type = band.DataType
    # from http://www.gdal.org/gdal_8h.html
    if 0 < data_type < 6:  # data_type 1:5 are int data types
        data_type = 'Categorical'
    else:
        data_type = 'Ordinal'
    return data_type


def get_numpy_stats(tif, writer):
    ds = gdal.Open(tif, gdal.GA_ReadOnly)
    number_of_bands = ds.RasterCount

    if ds.RasterCount > 1:
        log.info('Found multibanded geotif {}'.format(basename(tif)))
        for b in range(number_of_bands):
            write_rows(stats=number_of_bands(ds, tif, b + 1), writer=writer)
    else:
        log.info('Found single band geotif {}'.format(basename(tif)))
        write_rows(stats=number_of_bands(ds, tif, 1), writer=writer)


def numpy_band_stats(ds, tif, band_no, partitions=100):
    band = ds.GetRasterBand(band_no)
    data = band.ReadAsArray()
    no_data_val = band.GetNoDataValue()
    data_type = get_datatype(band)
    mask_data = ma.masked_where(data == no_data_val, data)

    if data_type is 'Categorical':
        no_categories = np.max(mask_data) - np.min(mask_data) + 1
    else:
        no_categories = np.nan

    image_source = geoio.RasterioImageSource(tif)
    l = [basename(tif), band_no, no_data_val,
         ds.RasterYSize, ds.RasterXSize,
         np.min(mask_data), np.max(mask_data),
         np.mean(mask_data), np.std(mask_data),
         data_type, float(no_categories),
         image_nans(image_source, partitions)]
    ds = None
    return [str(a) for a in l]


def image_nans(image_source, n_subchunks):

    result = []
    for subchunk_index in range(n_subchunks):
        result.append(
            chunk_nancount(image_source, n_subchunks, subchunk_index))
    return np.sum(result)


def chunk_nancount(image_source, n_subchunks, subchunk_index):
    r = features.extract_subchunks(image_source,
                                   subchunk_index,
                                   n_subchunks,
                                   patchsize=0)
    nan_count = np.isnan(r).sum()

    # if the r is entirely masked (due to nodata) maskedconstant is returned
    if isinstance(nan_count, np.ma.core.MaskedConstant):
        return 0
    else:
        return nan_count
