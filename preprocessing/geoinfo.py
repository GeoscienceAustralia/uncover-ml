import logging
import click
import glob
import numpy as np
from numpy import ma
from osgeo import gdal
from os.path import join, abspath, basename, isdir, isfile
import csv
import gc

log = logging.getLogger(__name__)


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('report_file')
def inspect(input_dir, report_file):
    input_dir = abspath(input_dir)
    if isdir(input_dir):
        log.info('Reading tifs from {}'.format(input_dir))
        tifs = glob.glob(join(input_dir, '*.tif'))
    else:
        log.info('Reporting geoinfo for {}'.format(input_dir))
        tifs = [input_dir]

    with open(report_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(['FineName', 'NoDataValue',
                        'rows', 'cols', 'Min', 'Max', 'Mean', 'Std'])
        for t in tifs:
            get_stats(t, writer)


def write_rows(stats, writer):
    writer.writerow(stats)


def get_stats(tif, writer):
    ds = gdal.Open(tif, gdal.GA_ReadOnly)
    number_of_bands = ds.RasterCount

    if ds.RasterCount > 1:
        log.info('Found multibanded geotif {}'.format(basename(tif)))
        for b in range(number_of_bands):
            write_rows(stats=band_stats(ds, tif, b + 1), writer=writer)
    else:
        log.info('Found single band geotif {}'.format(basename(tif)))
        write_rows(stats=band_stats(ds, tif, 1), writer=writer)


def band_stats(ds, tif, band_no):
    log.info('Calculating band stats for {} band {}'.format(tif, band_no))
    band = ds.GetRasterBand(band_no)
    # For statistics calculation
    stats = band.ComputeStatistics(False)
    no_data_val = band.GetNoDataValue()
    l = [basename(tif), no_data_val,
         ds.RasterYSize, ds.RasterXSize,
         stats[0], stats[1],
         stats[2], stats[3]]
    ds = None  # close dataset
    return [str(a) for a in l]


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
    mask_data = ma.masked_where(data == no_data_val, data)
    l = [basename(tif), no_data_val,
         ds.RasterYSize, ds.RasterXSize,
         np.min(mask_data), np.max(mask_data),
         np.mean(mask_data), np.std(mask_data)]
    ds = None
    return [str(a) for a in l]
