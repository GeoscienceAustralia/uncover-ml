from __future__ import print_function
"""
python utility to crop, resample, and mask, a larger ifg into samller ifg if
desired.

example usage:
"""
from optparse import OptionParser
import subprocess
from osgeo import gdal

TSRS = 'EPSG:3112'
EXTENTS = [str(s) for s in
            [-2362974.47956, -5097641.80634, -2362000.47956, -5097000.80634]]
          # [-2362974.47956, -5097641.80634, 2251415.52044, -1174811.80634]]
OUTPUT_RES = [str(s) for s in [90, 90]]


def crop_resample(input_file, output_file, sampling):
    """
    The -wm flag sets the amount of memory used for the
    big working buffers (the chunks) when warping.  But down within GDAL
    itself the format drivers use a block cache which comes from a seperate
    memory pool controlled by GDAL_CACHEMAX.
    Parameters
    ----------
    input_file: input tif
    output_file: output cropped and resampled tif
    sampling: sampling algo to use

    This can also be done by avoiding system calls using the python api.
    -------

    """
    # TODO: add extents checking between input_file and extents
    cmd = ['gdalwarp', '-overwrite'] + \
          ['-t_srs'] + [TSRS] + \
          ['-tr'] + OUTPUT_RES +  \
          ['-te'] + EXTENTS + \
          ['-r'] + [sampling] + \
          ['-wm'] + ['200'] + \
          ['--config', 'GDAL_CACHEMAX', '150']
    cmd += [input_file, output_file]
    subprocess.check_call(cmd)


def mask_data(output_tif, mask_file):
    mask_ds = gdal.Open(mask_file, gdal.GA_ReadOnly)
    mask_band = mask_ds.GetRasterBand(1)
    mask_no_data_value = mask_band.GetNoDataValue()
    masked_data = mask_band.ReadAsArray()
    mask_array = masked_data == mask_no_data_value
    mask_ds = None
    out_ds = gdal.Open(output_tif, gdal.GA_Update)
    out_band = out_ds.GetRasterBand(1)
    out_data = out_band.ReadAsArray()
    out_data[mask_array] = mask_no_data_value
    out_band.WriteArray(out_data)
    out_ds = None  # flush data to cache


if __name__ == '__main__':
    parser = OptionParser(usage='%prog -i input_file -o output_file'
                                ' -s sampling\n'
                                'Crop a larger interferrogram into '
                                'smaller ones')
    parser.add_option('-i', '--input', type=str, dest='input_file',
                      help='name of input interferrogram')

    parser.add_option('-o', '--out', type=str, dest='output_file',
                      help='name of cropped output interferrogram')

    parser.add_option('-m', '--mask', type=str, dest='mask_file',
                      help='name of mask file')

    parser.add_option('-s', '--sampling', type=str, dest='sampling',
                      help='sampling algorithm to use')

    options, args = parser.parse_args()

    if not options.input_file:  # if filename is not given
        parser.error('Input filename not given.')

    if not options.output_file:  # if filename is not given
        parser.error('Output filename not given.')

    if not options.sampling:  # if filename is not given
        options.sampling = 'near'

    crop_resample(input_file=options.input_file,
                  output_file=options.output_file,
                  sampling=options.sampling)

    if options.mask_file:  # if filename is not given
        mask_data(options.output_file, options.mask_file)


