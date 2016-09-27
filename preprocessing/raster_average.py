import numpy as np
from scipy import ndimage
import logging
import click
from os.path import abspath, join, basename
import glob
from osgeo import gdal, gdalconst
from subprocess import check_call

log = logging.getLogger(__name__)
COMMON = ['--config', 'GDAL_CACHEMAX', '200']
TILES = ['-co', 'TILED=YES']
TRANSLATE = 'gdal_translate'

@click.group()
def cli():
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument('input_dir')
@click.argument('out_dir')
@click.option('-s', '--size', type=int, default=3,
              help='size of the uniform filter to '
                   'perform uniform 2d average according to '
                   'scipy.ndimage.uniform_filter')
def average(input_dir, out_dir, size):
    input_dir = abspath(input_dir)
    log.info('Reading tifs from {}'.format(input_dir))
    tifs = glob.glob(join(input_dir, '*.tif'))

    for t in tifs:
        ds = gdal.Open(t, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        # data_type = gdal.GetDataTypeName(band.DataType)
        data = band.ReadAsArray()
        no_data_val = band.GetNoDataValue()
        averaged_data = filter_data(data, size, no_data_val)
        log.info('Calculated average for {}'.format(basename(t)))

        output_file = join(out_dir, 'average_' + basename(t))
        out_ds = gdal.GetDriverByName('GTiff').Create(
            output_file, ds.RasterXSize, ds.RasterYSize,
            1, band.DataType)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(averaged_data)
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_band.FlushCache()  # Write data to disc
        out_ds = None  # close out_ds
        ds = None  # close dataset

        log.info('Finished converting {}'.format(basename(t)))


def filter_data(data, size, no_data_val=None):
    """
    This does not work with masked array.
    ndimage.uniform_filter does not respect masked array
    Parameters
    ----------
    data
    size
    no_data_val

    Returns
    -------

    """
    if no_data_val:
        mask = data == no_data_val
        data[mask] = np.nan
    averaged_data = np.zeros_like(data)
    ndimage.uniform_filter(data,
                           output=averaged_data,
                           size=size,
                           mode='reflect')
    return averaged_data


