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
           [-2362974.47956, -5097641.80634, 2251415.52044, -1174811.80634]]
OUTPUT_RES = [str(s) for s in [90, 90]]


def crop_resample(input_file, output_file, sampling):
    # TODO: add extents checking between input_file and extents
    cmd = ['gdalwarp', '-overwrite'] + \
          ['-t_srs'] + [TSRS] + \
          ['-tr'] + OUTPUT_RES +  \
          ['-te'] + EXTENTS + \
          ['-r'] + [sampling] + \
          ['-wm'] + ['20']  # use 20MB Cache
    cmd += [input_file, output_file]
    subprocess.check_call(cmd)


if __name__ == '__main__':
    parser = OptionParser(usage='%prog -i input_file -o output_file'
                                ' -s sampling\n'
                                'Crop a larger interferrogram into '
                                'smaller ones')
    parser.add_option('-i', '--input', type=str, dest='input_file',
                      help='name of input interferrogram')

    parser.add_option('-o', '--out', type=str, dest='output_file',
                      help='name of cropped output interferrogram')

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


