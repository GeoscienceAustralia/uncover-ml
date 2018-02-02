"""
python utility to crop a geotif

example usage:
# python crop_gtif.py -i input.tif -o out.tif -e
    '150.91 -34.229999976 150.949166651  -34.17'

`-e extents`
extents = 'lowerleft.x, lowerleft.y upperright.x upperright.y'
"""
from __future__ import print_function
from optparse import OptionParser
import subprocess


def crop_using_gdalwarp(input_file, output_file, extents):
    # TODO: add extents checking between input_file and extents
    print('Cropping {} into {}'.format(input_file, output_file))
    extents_str = [str(e) for e in extents]
    cmd = ['gdalwarp', '-overwrite', '-q', '-te'] + extents_str
    cmd += [input_file, output_file]
    subprocess.check_call(cmd)


if __name__ == '__main__':
    parser = OptionParser(usage='%prog -i input_file -o output_file'
                                ' -e extents\n'
                                'Crop a larger geotifs into '
                                'smaller ones.\n'
                                'extents = lowerleft.x, lowerleft.y'
                                'upperright.x upperright.y')

    parser.add_option('-i', '--input', type=str, dest='input_file',
                      help='name of input geotif')

    parser.add_option('-o', '--out', type=str, dest='output_file',
                      help='name of cropped output geotif')

    parser.add_option('-e', '--extents', type=str, dest='extents',
                      help='extents to be used for the cropped file.\n'
                           'needs to be a list of 4 floats with spaces\n'
                           'example: '
                           "-e '150.91 -34.229999976 150.949166651 -34.17'")

    options, args = parser.parse_args()

    if not options.input_file:  # if filename is not given
        parser.error('Input filename not given.')

    if not options.output_file:  # if filename is not given
        parser.error('Output filename not given.')

    if not options.extents:  # if filename is not given
        parser.error('Crop extents must be provided')

    extents = [float(t) for t in options.extents.split()]

    if len(extents) != 4:
        raise AttributeError('extents to be used for the cropped file.\n'
                             'needs to be a list or tuples of 4 floats\n'
                             "example:"
                             "--extents "
                             "'150.91 -34.229999976 150.949166651 -34.17'")

    crop_using_gdalwarp(input_file=options.input_file,
                        output_file=options.output_file,
                        extents=extents)

