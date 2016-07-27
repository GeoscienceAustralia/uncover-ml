from __future__ import print_function
"""
python utility to crop, resample, reproject and optionally mask, a larger ifg into smaller ifg if desired.
example usage: python crop_mask_resample_reproject.py -i slope_fill2.tif -o slope_fill2_out.tif -e '-2362974.47956 -5097641.80634 2251415.52044 -1174811.80634' -m mack_LCC.tif -s bilinear

Also outputs a jpeg file corresponding to the final cropped, resampled, reprojectd and masked geotiff.
"""
from optparse import OptionParser
import subprocess
from osgeo import gdal
import os

TSRS = 'EPSG:3112'
OUTPUT_RES = [str(s) for s in [90, 90]]
TMP_OUT = 'tmp_out.tif'
TMP_VRT = 'tmp.vrt'


def crop_mask_resample(input_file, output_file, sampling, extents):
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
    cmd = ['gdalwarp', '-overwrite', '-multi'] + \
          ['-t_srs'] + [TSRS] + \
          ['-tr'] + OUTPUT_RES +  \
          ['-te'] + extents + \
          ['-r'] + [sampling] + \
          ['-wm'] + ['100'] + \
          ['--config', 'GDAL_CACHEMAX', '150']
    cmd += [input_file, output_file]
    subprocess.check_call(cmd)


def apply_mask(mask_file, output_file, jpeg):
    """
    Parameters
    ----------
    mask_file: mask file path
    output_file: output geotiff path
    jpeg: boolean, whether to produce jpeg or not
    -------

    """

    cmd_build = ['gdalbuildvrt', '-separate',
                 '--config', 'GDAL_CACHEMAX', '150',
                 TMP_VRT,
                 TMP_OUT,
                 mask_file
                 ]
    subprocess.check_call(cmd_build)

    cmd_mask = ['gdal_translate', '-b', '1', '-mask', '2',
                '--config', 'GDAL_CACHEMAX', '150',
                TMP_VRT,
                output_file]
    subprocess.check_call(cmd_mask)
    # clean up
    os.remove(TMP_VRT)
    os.remove(TMP_OUT)

    if jpeg:
        dir_name = os.path.dirname(output_file)
        jpeg_file = os.path.basename(output_file).split('.')[0] + '.jpg'
        jpeg_file = os.path.join(dir_name, jpeg_file)
        cmd_jpg = ['gdal_translate', '-of', 'JPEG', '-scale',
                   output_file,
                   jpeg_file]
        subprocess.check_call(cmd_jpg)
        print('created jpg')


if __name__ == '__main__':
    parser = OptionParser(usage='%prog -i input_file -o output_file'
                                ' -e extents\n'
                                ' -s sampling (optional)\n'
                                ' -m mask_file (optional)\n'
                                'Crop, resample, reproject and '
                                'optionally mask a larger '
                                'interferrogram into smaller ones')
    parser.add_option('-i', '--input', type=str, dest='input_file',
                      help='name of input interferrogram')

    parser.add_option('-o', '--out', type=str, dest='output_file',
                      help='name of cropped output interferrogram')

    parser.add_option('-m', '--mask', type=str, dest='mask_file',
                      help='name of mask file')

    parser.add_option('-e', '--extents', type=str, dest='extents',
                      help='extents to be used for the cropped file.\n'
                           'needs to be a list of 4 floats with spaces\n'
                           'example: '
                           "-e '150.91 -34.229999976 150.949166651 -34.17'")
    parser.add_option('-s', '--sampling', type=str, dest='sampling',
                      help='sampling algorithm to use')

    options, args = parser.parse_args()

    if not options.input_file:  # if filename is not given
        parser.error('Input filename not given.')

    if not options.output_file:  # if filename is not given
        parser.error('Output filename not given.')

    if not options.sampling:  # if filename is not given
        options.sampling = 'near'

    if not options.extents:  # if filename is not given
        parser.error('Crop extents must be provided')

    extents = [float(t) for t in options.extents.split()]

    if len(extents) != 4:
        raise AttributeError('extents to be used for the cropped file.\n'
                             'needs to be a list or tuples of 4 floats\n'
                             "example:"
                             "--extents '-2362974.47956 -5097641.80634 "
                             "2251415.52044 -1174811.80634'")
    options.extents = [str(s) for s in extents]
    crop_mask_resample(input_file=options.input_file,
                       output_file=TMP_OUT,
                       sampling=options.sampling,
                       extents=options.extents)

    if options.mask_file:
        apply_mask(mask_file=options.mask_file,
                   output_file=options.output_file,
                   jpeg=True)



