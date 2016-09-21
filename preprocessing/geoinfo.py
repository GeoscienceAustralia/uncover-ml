import logging
import click
import glob
import numpy as np
from numpy import ma
from osgeo import gdal
from os.path import join, abspath, basename
import csv
import gc

log = logging.getLogger(__name__)


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument('input_dir')
@click.argument('report_file')
def inspect(input_dir, report_file):
    input_dir = abspath(input_dir)
    log.info('Reading tifs from {}'.format(input_dir))
    tifs = glob.glob(join(input_dir, '*.tif'))

    with open(report_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(['FineName', 'NoDataValue',
                        'rows', 'cols', 'Min', 'Max', 'Mean', 'Std'])
        for t in tifs:
            write_rows(t, writer)


def write_rows(t, writer):
    writer.writerow(get_stats(t))
    gc.collect()


def get_stats(t):
    ds = gdal.Open(t, gdal.GA_ReadOnly)
    if ds.RasterCount == 3:
        log.info('Found multibanded geotif {}'.format(basename(t)))
        log.info('Please inspect bands individually')
    else:
        log.info('Found single band geotif {}'.format(basename(t)))

    band = ds.GetRasterBand(1)

    # For statistics calculation
    stats = band.ComputeStatistics(False)
    no_data_val = band.GetNoDataValue()
    l = [basename(t), no_data_val,
         ds.RasterYSize, ds.RasterXSize,
         stats[0], stats[1],
         stats[2], stats[3]]
    ds = None  # close dataset
    return [str(a) for a in l]


def get_numpy_stats(t):
    ds = gdal.Open(t, gdal.GA_ReadOnly)
    if ds.RasterCount == 3:
        log.info('Found multibanded geotif {}'.format(basename(t)))
        log.info('Please inspect bands individually')
    else:
        log.info('Found single band geotif {}'.format(basename(t)))
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    no_data_val = band.GetNoDataValue()
    mask_data = ma.masked_where(data == no_data_val, data)
    l = [basename(t), no_data_val,
         ds.RasterYSize, ds.RasterXSize,
         np.min(mask_data), np.max(mask_data),
         np.mean(mask_data), np.std(mask_data)]
    ds = None
    return [str(a) for a in l]
