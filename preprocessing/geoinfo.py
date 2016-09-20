import logging
import click
from osgeo import gdal
import glob
from os.path import join, abspath, basename
import numpy as np
import csv

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
                        'rows', 'cols', 'Max', 'Min', 'Mean', 'Std'])
        for t in tifs:
            write_rows(t, writer)


def write_rows(t, writer):
    ds = gdal.Open(t, gdal.GA_ReadOnly)
    if ds.RasterCount == 3:
        log.info('Found multibanded geotif {}'.format(basename(t)))
        log.info('Please inspect bands individually')
    else:
        log.info('Found single band geotif {}'.format(basename(t)))
    l = get_stats(ds, t)
    writer.writerow([str(a) for a in l])
    ds = None   # close dataset


def get_stats(ds, t):
    data = ds.ReadAsArray()
    band = ds.GetRasterBand(1)
    no_data_val = band.GetNoDataValue()
    l = [basename(t), no_data_val,
         ds.RasterYSize, ds.RasterXSize,
         np.max(data), np.min(data),
         np.mean(data), np.std(data)]
    return l


