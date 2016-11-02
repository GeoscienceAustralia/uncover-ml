import logging
import click
import glob
import numpy as np
from numpy import ma
from osgeo import gdal
from os.path import join, abspath, basename, isdir, isfile
import csv
from uncoverml import mpiops

log = logging.getLogger(__name__)


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
@click.option('-e', '--extension',
              type=str, default='tif',
              help='geotif extension, e.g., `tif`, or `tiff`')
def inspect(input_dir, report_file, extension):
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
                         'DataType', 'Categories'])
        process_tifs = np.array_split(tifs, mpiops.chunks)[mpiops.chunk_index]

        stats = []  # process geotiff stats including multibanded geotif
        for t in process_tifs:
            stats.append(get_stats(t))

        # gather all process geotif stats in stats dict
        stats = _join_dicts(stats)

        # global gather in root
        stats = _join_dicts(mpiops.comm.gather(stats, root=0))

        if mpiops.chunk_index == 0:
            for k, v in stats.items():
                write_rows(v, writer)


def write_rows(stats, writer):
    writer.writerow(stats)


def get_stats(tif):
    ds = gdal.Open(tif, gdal.GA_ReadOnly)
    number_of_bands = ds.RasterCount

    if ds.RasterCount > 1:
        d = {}
        log.info('Found multibanded geotif {}'.format(basename(tif)))
        for b in range(number_of_bands):
            d['{tif}_{b}'.format(tif=tif, b=b)] = band_stats(ds, tif, b + 1)
        return d
    else:
        log.info('Found single band geotif {}'.format(basename(tif)))
        return {tif: band_stats(ds, tif, 1)}


def band_stats(ds, tif, band_no):
    log.info('Calculating band stats for {} band {}'.format(tif, band_no))
    band = ds.GetRasterBand(band_no)
    # For statistics calculation
    stats = band.ComputeStatistics(False)
    no_data_val = band.GetNoDataValue()
    data_type = get_datatype(band)

    if data_type is 'Categorical':
        no_categories = stats[1] - stats[0] + 1
    else:
        no_categories = np.nan

    l = [basename(tif), band_no, no_data_val,
         ds.RasterYSize, ds.RasterXSize,
         stats[0], stats[1],
         stats[2], stats[3],
         data_type, no_categories]
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


def numpy_band_stats(ds, tif, band_no):
    band = ds.GetRasterBand(band_no)
    data = band.ReadAsArray()
    no_data_val = band.GetNoDataValue()
    data_type = get_datatype(band)

    mask_data = ma.masked_where(data == no_data_val, data)

    if data_type is 'Categorical':
        no_categories = np.max(mask_data) - np.min(mask_data) + 1
    else:
        no_categories = np.nan

    l = [basename(tif), band_no, no_data_val,
         ds.RasterYSize, ds.RasterXSize,
         np.min(mask_data), np.max(mask_data),
         np.mean(mask_data), np.std(mask_data),
         data_type, no_categories]
    ds = None
    return [str(a) for a in l]
