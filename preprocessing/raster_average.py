import warnings
from subprocess import check_call
import shutil
import glob
import logging
from os.path import abspath, join, basename, isdir, isfile
import click
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage
from osgeo import gdal
from uncoverml import mpiops

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)
COMMON = ['--config', 'GDAL_CACHEMAX', '200']
TILES = ['-co', 'TILED=YES']
TRANSLATE = 'gdal_translate'
func_map = {'nanmean': np.nanmean,
            'nanmax': np.nanmax,
            'nanmin': np.nanmin,
            'nanmedian': np.nanmedian}


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

    for tif in tifs:
        data_source = gdal.Open(tif, gdal.GA_ReadOnly)
        band = data_source.GetRasterBand(1)
        # data_type = gdal.GetDataTypeName(band.DataType)
        data = band.ReadAsArray()
        no_data_val = band.GetNoDataValue()
        averaged_data = filter_data(data, size, no_data_val)
        log.info('Calculated average for {}'.format(basename(tif)))

        output_file = join(out_dir, 'average_' + basename(tif))
        out_ds = gdal.GetDriverByName('GTiff').Create(
            output_file, data_source.RasterXSize, data_source.RasterYSize,
            1, band.DataType)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(averaged_data)
        out_ds.SetGeoTransform(data_source.GetGeoTransform())
        out_ds.SetProjection(data_source.GetProjection())
        out_band.FlushCache()  # Write data to disc
        out_ds = None  # close out_ds
        data_source = None  # close dataset

        log.info('Finished converting {}'.format(basename(tif)))


@cli.command()
@click.argument('input_dir')
@click.argument('out_dir')
@click.option('-s', '--size', type=int, default=5,
              help='size of the uniform filter to '
                   'perform uniform 2d average according to '
                   'scipy.ndimage.uniform_filter')
def gdalaverage(input_dir, out_dir, size):
    """
    average data using gdal's averaging method.
    Parameters
    ----------
    input_dir: str
        input dir name of the tifs that needs to be averaged
    out_dir: str
        output dir name
    size: int, optional
        size of kernel
    Returns
    -------

    """
    input_dir = abspath(input_dir)
    log.info('Reading tifs from {}'.format(input_dir))
    tifs = glob.glob(join(input_dir, '*.tif'))

    process_tifs = np.array_split(tifs, mpiops.chunks)[mpiops.chunk_index]

    for tif in process_tifs:
        data_set = gdal.Open(tif, gdal.GA_ReadOnly)
        # band = data_set.GetRasterBand(1)
        # data_type = gdal.GetDataTypeName(band.DataType)
        # data = band.ReadAsArray()
        # no_data_val = band.GetNoDataValue()
        # averaged_data = filter_data(data, size, no_data_val)
        log.info('Calculated average for {}'.format(basename(tif)))

        output_file = join(out_dir, 'average_' + basename(tif))
        src_gt = data_set.GetGeoTransform()
        tmp_file = '/tmp/tmp_{}.tif'.format(mpiops.chunk_index)
        resample_cmd = [TRANSLATE] + [tif, tmp_file] + \
            ['-tr', str(src_gt[1]*size), str(src_gt[1]*size)] + \
            ['-r', 'bilinear']
        check_call(resample_cmd)
        rollback_cmd = [TRANSLATE] + [tmp_file, output_file] + \
            ['-tr', str(src_gt[1]), str(src_gt[1])]
        check_call(rollback_cmd)
        log.info('Finished converting {}'.format(basename(tif)))


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


def filter_center(data, size=3, no_data_val=None, func=np.nanmean,
                  mask_no_data=False):
    """
    Parameters
    ----------
    data: input data
    size: odd number uniform filtering kernel size
    no_data_val: value in matrix that is treated as no data value
    func: function to use, choose from np.nanmean/median/max/min etc.
    mask_no_data: bool, if True will keep the original no data pixel intact

    Returns: nanmean of the matrix A filtered by a uniform kernel of size=size
    -------
    Adapted from: http://stackoverflow.com/questions/23829097/python-numpy-fastest-method-for-2d-kernel-rank-filtering-on-masked-arrays-and-o?rq=1

    Notes
    -----
    This function `centers` the kernel at the target pixel.
    This is slightly different from scipy.ndimage.uniform_filter application.
    In scipy.ndimage.uniform_filter, a convolution approach is implemented.
    An equivalent is scipy.ndimage.uniform_filter like convolution approach
    with no_data_val/nan handling can be found in
    filter_broadcast_uniform_filter in this module.

    Change function to nanmedian, nanmax, nanmin as required.
    """

    assert size % 2 == 1, 'Please supply an odd size'
    rows, cols = data.shape

    padded_data = np.empty(shape=(rows + size-1,
                                  cols + size-1),
                           dtype=data.dtype)
    padded_data[:] = np.nan
    rows_pad, cols_pad = padded_data.shape

    if no_data_val is not None:
        mask = data == no_data_val
        data[mask] = np.nan

    padded_data[size//2:rows_pad - size//2,
                size//2: cols_pad - size//2] = data.copy()

    row, col = data.shape

    stride_data = as_strided(padded_data, (row, col, size, size),
                             padded_data.strides+padded_data.strides)
    stride_data = stride_data.copy().reshape((row, col, size**2))

    avg = func(stride_data, axis=2)
    avg[np.isnan(avg)] = no_data_val

    if mask_no_data:
        avg[mask] = no_data_val

    return avg


def filter_uniform_filter(data, size=3, no_data_val=None,
                          func=np.nanmean):
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
    rows, cols = data.shape

    padded_A = np.empty(shape=(rows + size-1,
                               cols + size-1),
                        dtype=data.dtype)
    padded_A[:] = np.nan
    rows_pad, cols_pad = padded_A.shape

    if no_data_val:
        mask = data == no_data_val
        data[mask] = np.nan

    padded_A[size-1: rows_pad, size - 1: cols_pad] = data.copy()

    n, m = data.shape
    strided_data = as_strided(padded_A, (n, m, size, size),
                              padded_A.strides+padded_A.strides)
    strided_data = strided_data.copy().reshape((n, m, size**2))

    return func(strided_data, axis=2)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path(exists=True))
@click.option('-f', '--func',
              type=click.Choice(['nanmean', 'nanmedian',
                                 'nanmax', 'nanmin']),
              default='nanmean', help='Level of logging')
@click.option('-p', '--partitions', type=int, default=1,
              help='Number of partitions for calculating 2d statistics')
@click.option('-s', '--size', type=int, default=3,
              help='size of the uniform filter to '
                   'calculate 2d stats with the uniform kernel '
                   'centered around the target pixel for continuous data. '
                   'Categorical data are copied unchanged.')
@click.option('-m', '--mask', type=bool, default=False,
              help='whether to keep the original no data pixels intact')
def mean(input_dir, out_dir, size, func, partitions, mask):
    input_dir = abspath(input_dir)
    if isdir(input_dir):
        log.info('Reading tifs from {}'.format(input_dir))
        tifs = glob.glob(join(input_dir, '*.tif'))
    else:
        assert isfile(input_dir)
        tifs = [input_dir]

    process_tifs = np.array_split(tifs, mpiops.chunks)[mpiops.chunk_index]

    for tif in process_tifs:
        log.info('Starting to average {}'.format(basename(tif)))
        treat_file(tif, out_dir, size, func, partitions, mask)
        log.info('Finished averaging {}'.format(basename(tif)))


def treat_file(tif, out_dir, size, func, partitions, mask_no_data=False):
    """
    Parameters
    ----------
    tif: input geotif
    out_dir: output dir
    size: odd int (2n+1)
        size of kernel, has to be odd
    func: str
        one of nanmean, nanmedian, nanmax, nanmin
    partitions: int
        number of partitions for calculating 2d statistics
    mask_no_data: bool
        whether to keep the original nodatavalues intact

    Returns
    -------
        None
    """
    data_source = gdal.Open(tif, gdal.GA_ReadOnly)
    band = data_source.GetRasterBand(1)
    no_data_val = band.GetNoDataValue()
    output_file = join(out_dir, basename(tif))
    if no_data_val is None:
        log.error('NoDataValue was not found in input image {} \n'
                  'and this file was skipped'.format(basename(tif)))
        return
    if band.DataType <= 4:
        shutil.copy(tif, output_file)
        data_source = None
        return
    out_ds = gdal.GetDriverByName('GTiff').Create(
        output_file, data_source.RasterXSize, data_source.RasterYSize,
        1, band.DataType)
    out_band = out_ds.GetRasterBand(1)

    tif_rows = data_source.RasterYSize
    partition_rows = np.array_split(range(tif_rows), partitions)

    xoff = 0
    win_xsize = data_source.RasterXSize
    pad_width = size // 2

    for p in range(partitions):
        rows = partition_rows[p]

        # grab data with pad_width added appropriately
        _ysize, win_ysize, yoff = _edge_adjust(p, pad_width, partitions, rows)

        data = band.ReadAsArray(xoff=xoff, yoff=yoff,
                                win_xsize=win_xsize, win_ysize=win_ysize)

        averaged_data = filter_center(data, size, no_data_val, func_map[func],
                                      mask_no_data)

        # discard pad_width
        averaged_data = averaged_data[_ysize: len(rows) + _ysize]

        out_band.WriteArray(averaged_data,
                            xoff=0, yoff=int(rows[0]))
        out_band.FlushCache()  # Write data to disc
        log.info('Calculated average for {} partition {}'.format(
            basename(tif), p))

    out_band.SetNoDataValue(no_data_val)
    out_ds.SetGeoTransform(data_source.GetGeoTransform())
    out_ds.SetProjection(data_source.GetProjection())

    out_ds = None  # close out_ds
    band = None
    data_source = None  # close dataset


def _edge_adjust(partition, pad_width, partitions, rows):
    # The following if else is to make sure we are not having
    # partition and mpi splitting effects
    # when p=0 don't look back
    if partition == 0:
        yoff = int(rows[0])
        win_ysize = len(rows) + pad_width
        _ysize = 0
    elif partition == partitions - 1:
        yoff = int(rows[0]) - pad_width
        win_ysize = len(rows) + pad_width
        _ysize = pad_width
    else:
        yoff = int(rows[0]) - pad_width
        win_ysize = len(rows) + pad_width * 2
        _ysize = pad_width
    if partitions == 1:
        yoff = int(rows[0])
        win_ysize = len(rows)
        _ysize = 0

    return _ysize, win_ysize, yoff
