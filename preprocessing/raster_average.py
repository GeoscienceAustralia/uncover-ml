import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage
import logging
import click
from os.path import abspath, join, basename, isdir, isfile
import shutil
import glob
from osgeo import gdal, gdalconst
from subprocess import check_call
from PIL import ImageFilter, Image
from uncoverml import mpiops


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


@cli.command()
@click.argument('input_dir')
@click.argument('out_dir')
@click.option('-s', '--size', type=int, default=5,
              help='size of the uniform filter to '
                   'perform uniform 2d average according to '
                   'scipy.ndimage.uniform_filter')
def gdalaverage(input_dir, out_dir, size):
    input_dir = abspath(input_dir)
    log.info('Reading tifs from {}'.format(input_dir))
    tifs = glob.glob(join(input_dir, '*.tif'))

    process_tifs = np.array_split(tifs, mpiops.chunks)[mpiops.chunk_index]

    for t in process_tifs:
        ds = gdal.Open(t, gdal.GA_ReadOnly)
        # band = ds.GetRasterBand(1)
        # data_type = gdal.GetDataTypeName(band.DataType)
        # data = band.ReadAsArray()
        # no_data_val = band.GetNoDataValue()
        # averaged_data = filter_data(data, size, no_data_val)
        log.info('Calculated average for {}'.format(basename(t)))

        output_file = join(out_dir, 'average_' + basename(t))
        src_gt = ds.GetGeoTransform()
        tmp_file = '/tmp/tmp_{}.tif'.format(mpiops.chunk_index)
        resample_cmd = [TRANSLATE] + [t, tmp_file] + \
            ['-tr', str(src_gt[1]*size), str(src_gt[1]*size)] + \
            ['-r', 'bilinear']
        check_call(resample_cmd)
        rollback_cmd = [TRANSLATE] + [tmp_file, output_file] + \
            ['-tr', str(src_gt[1]), str(src_gt[1])]
        check_call(rollback_cmd)
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
                           mode='nearest')
    return averaged_data


def filter_center(A, size=3, no_data_val=None, func=np.nanmean):
    """
    Parameters
    ----------
    A = input data
    size = odd number uniform filtering kernel size
    no_data_val = value in matrix that is treated as no data value
    func: function to use, choose from np.nanmean/median/max/min etc.

    Returns: nanmean of the matrix A filtered by a uniform kernel of size=size
    -------
    Adapted from: http://stackoverflow.com/questions/23829097/python-numpy-fastest-method-for-2d-kernel-rank-filtering-on-masked-arrays-and-o?rq=1

    Notes:
    This function `centers` the kernel at the target pixel.
    This is slightly different from scipy.ndimage.uniform_filter application.
    In scipy.ndimage.uniform_filter, a convolution approach is implemented.
    An equivalent is scipy.ndimage.uniform_filter like convolution approach with
    no_data_val/nan handling can be found in filter_broadcast_uniform_filter in
    this module.

    Change function to nanmeadian, nanmax, nanmin as required.
    """

    assert size % 2 == 1, 'Please supply an odd size'
    rows, cols = A.shape

    padded_A = np.empty(shape=(rows + size-1,
                               cols + size-1),
                        dtype=A.dtype)
    padded_A[:] = np.nan
    rows_pad, cols_pad = padded_A.shape

    if no_data_val:
        mask = A == no_data_val
        A[mask] = np.nan

    padded_A[size//2:rows_pad - size//2, size//2: cols_pad - size//2] = A.copy()

    N = A.shape[0]
    B = as_strided(padded_A, (N, N, size, size),
                   padded_A.strides+padded_A.strides)
    B = B.copy().reshape((N, N, size**2))
    return func(B, axis=2)


def filter_broadcast_uniform_filter(A, size=3, no_data_val=None):
    """
    Parameters
    ----------
    A = input data
    size = odd number uniform filtering kernel size
    no_data_val = value in matrix that is treated as no data value

    Returns: nanmean of the matrix A filtered by a uniform kernel of size=size
    -------
    Adapted from: http://stackoverflow.com/questions/23829097/python-numpy-fastest-method-for-2d-kernel-rank-filtering-on-masked-arrays-and-o?rq=1

    Notes:
    This is equivalent to scipy.ndimage.uniform_filter, but can handle nan's,
    and can use numpy nanmean/median/max/min functions.

    no_data_val/nan handling can be found in filter_broadcast_uniform_filter in
    this module.

    Change function to nanmeadian, nanmax, nanmin as required.
    """

    assert size % 2 == 1, 'Please supply an odd size'
    rows, cols = A.shape

    padded_A = np.empty(shape=(rows + size-1,
                               cols + size-1),
                        dtype=A.dtype)
    padded_A[:] = np.nan
    rows_pad, cols_pad = padded_A.shape

    if no_data_val:
        mask = A == no_data_val
        A[mask] = np.nan

    padded_A[size-1: rows_pad, size - 1: cols_pad] = A.copy()

    N = A.shape[0]
    B = as_strided(padded_A, (N, N, size, size),
                   padded_A.strides+padded_A.strides)
    B = B.copy().reshape((N, N, size**2))

    return np.nanmean(B, axis=2)


@cli.command()
@click.argument('input_dir')
@click.argument('out_dir')
@click.option('-s', '--size', type=int, default=3,
              help='size of the uniform filter to '
                   'perform 2d average with the uniform kernel '
                   'centered around the target pixel for continuous data. '
                   'Categorical data are copied unchanged.')
def mean(input_dir, out_dir, size):
    input_dir = abspath(input_dir)
    if isdir(input_dir):
        log.info('Reading tifs from {}'.format(input_dir))
        tifs = glob.glob(join(input_dir, '*.tif'))
    else:
        assert isfile(input_dir)
        tifs = [input_dir]

    process_tifs = np.array_split(tifs, mpiops.chunks)[mpiops.chunk_index]

    for t in process_tifs:
        log.info('Starting to average {}'.format(basename(t)))
        ds = gdal.Open(t, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray().astype(np.float32)
        no_data_val = band.GetNoDataValue()
        if not no_data_val:
            log.debug('NoDataValue was not found in input image {} \n'
                      'and this file was skipped'.format(basename(t)))
            continue
        output_file = join(out_dir, basename(t))
        if band.DataType == 1:
            shutil.copy(t, output_file)
            ds = None
            continue
        else:
            func = np.nanmean
        averaged_data = filter_center(data, size, no_data_val, func)

        log.info('Calculated average for {}'.format(basename(t)))

        out_ds = gdal.GetDriverByName('GTiff').Create(
            output_file, ds.RasterXSize, ds.RasterYSize,
            1, band.DataType)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(averaged_data)
        out_band.SetNoDataValue(no_data_val)
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_band.FlushCache()  # Write data to disc
        out_ds = None  # close out_ds
        ds = None  # close dataset

        log.info('Finished averaging {}'.format(basename(t)))
