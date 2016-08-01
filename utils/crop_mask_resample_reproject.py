from __future__ import print_function
"""
python utility to crop, resample, reproject and optionally mask, a larger ifg into smaller ifg if desired.
example usage: python crop_mask_resample_reproject.py -i slope_fill2.tif -o slope_fill2_out.tif -e '-2362974.47956 -5097641.80634 2251415.52044 -1174811.80634' -m mack_LCC.tif -r bilinear -j 0
Also optionally outputs a jpeg file corresponding to the final cropped, resampled, reprojectd and masked geotiff.
For resampling options see gdal.org/gdalwarp.html.
"""
from optparse import OptionParser
import subprocess
from osgeo import gdal
import os
import tempfile
import shutil
import gc

TSRS = 'EPSG:3112'
OUTPUT_RES = [str(s) for s in [90, 90]]
MASK_VALUES_TO_KEEP = 1
COMMON = ['--config', 'GDAL_CACHEMAX', '200']
TILES = ['-co', 'TILED=YES']

if 'PBS_JOBFS' in os.environ:
    TMPDIR = os.environ['PBS_JOBFS']
else:
    TMPDIR = tempfile.mkdtemp()


def crop_reproject_resample(input_file, output_file, sampling, extents):
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
    print('Crop/resample/reproject {}'.format(input_file))
    cmd = ['gdalwarp', '-overwrite', '-multi'] + \
          COMMON + TILES + \
          ['-t_srs'] + [TSRS] + \
          ['-tr'] + OUTPUT_RES +  \
          ['-te'] + extents + \
          ['-r'] + [sampling] + \
          ['-wm'] + ['200']
    cmd += [input_file, output_file]
    subprocess.check_call(cmd)


def apply_mask(mask_file, tmp_output_file, output_file, jpeg):
    """
    Parameters
    ----------
    mask_file: mask file path
    tmp_output_file: intermediate cropped geotiff before mask application
    output_file: output geotiff path
    jpeg: boolean, whether to produce jpeg or not
    -------

    """
    mask_ds = gdal.Open(mask_file, gdal.GA_ReadOnly)
    mask_data = mask_ds.GetRasterBand(1).ReadAsArray()
    mask = mask_data != MASK_VALUES_TO_KEEP
    del(mask_data)
    gc.collect()
    mask_ds = None  # close dataset

    out_ds = gdal.Open(tmp_output_file, gdal.GA_Update)
    out_band = out_ds.GetRasterBand(1)
    out_data = out_band.ReadAsArray()
    no_data_value = out_band.GetNoDataValue()
    if no_data_value:
        out_data[mask] = no_data_value
    else:
        print('NoDataValue was not set for {}'.format(output_file))
    out_band.WriteArray(out_data)
    out_ds = None  # close dataset and flush cache

    # copy file to output file
    shutil.copy(tmp_output_file, output_file)
    print('created', output_file)
    if jpeg:
        dir_name = os.path.dirname(output_file)
        jpeg_file = os.path.basename(output_file).split('.')[0] + '.jpg'
        jpeg_file = os.path.join(dir_name, jpeg_file)
        cmd_jpg = ['gdal_translate', '-ot', 'Byte', '-of', 'JPEG', '-scale',
                   tmp_output_file,
                   jpeg_file] + COMMON
        subprocess.check_call(cmd_jpg)
        print('created', jpeg_file)


def do_work(input_file, mask_file, output_file, resampling, extents, jpeg):
    # if we are going to use the mask, create the intermediate output
    # file locally, else create the final output file
    # also create the cropped mask file here, instead of inside apply_mask so
    # this mask cropping is not repeated in a batch run when using apply mask
    if mask_file:
        temp_output_file = tempfile.mktemp(suffix='.tif', dir=TMPDIR)
        cropped_mask_file = tempfile.mktemp(suffix='.tif', dir=TMPDIR)

        # crop/reproject/resample the mask
        crop_reproject_resample(mask_file, cropped_mask_file,
                                sampling='near',
                                extents=extents)

        # crop/reproject/resample the geotif
        crop_reproject_resample(input_file=input_file,
                                output_file=temp_output_file,
                                sampling=resampling,
                                extents=extents)

        # apply mask and optional y convert to jpeg
        apply_mask(mask_file=cropped_mask_file,
                   tmp_output_file=temp_output_file,
                   output_file=output_file,
                   jpeg=jpeg)
        # clean up
        os.remove(cropped_mask_file)
        print('removed intermediate cropped mask file', cropped_mask_file)
        os.remove(temp_output_file)
        print('removed intermediate cropped output file', temp_output_file)
    else:
        crop_reproject_resample(input_file=input_file,
                                output_file=output_file,
                                sampling=resampling,
                                extents=extents)


if __name__ == '__main__':
    parser = OptionParser(usage='%prog -i input_file -o output_file'
                                ' -e extents\n'
                                ' -r resampling (optional)\n'
                                ' -m mask_file (optional)\n'
                                ' -j jpeg_file (optional)\n'
                                'Crop, resample, reproject and '
                                'optionally mask a larger '
                                'interferrogram into smaller ones.'
                                'Optionally, create a outout jpeg')
    parser.add_option('-i', '--input', type=str, dest='input_file',
                      help='name of input tif')

    parser.add_option('-o', '--out', type=str, dest='output_file',
                      help='name of cropped output tif')

    parser.add_option('-m', '--mask', type=str, dest='mask_file',
                      help='name of mask file')

    parser.add_option('-e', '--extents', type=str, dest='extents',
                      help='extents to be used for the cropped file.\n'
                           'needs to be a list of 4 floats with spaces\n'
                           'example: '
                           "-e '150.91 -34.229999976 150.949166651 -34.17'")
    parser.add_option('-r', '--resampling', type=str, dest='resampling',
                      help='optional resampling algorithm to use')

    parser.add_option('-j', '--jpeg', type=int, dest='jpeg',
                      help='optional jpeg conversion. 0=no jpeg, 1=jpeg')

    options, args = parser.parse_args()

    if not options.input_file:  # if filename is not given
        parser.error('Input filename not given.')

    if not options.output_file:  # if filename is not given
        parser.error('Output filename not given.')

    if not options.resampling:  # if sampling is not given
        options.resampling = 'near'

    if not options.extents:  # if extents is not given
        parser.error('Crop extents must be provided')

    if not options.jpeg:  # if jpeg is not given
        options.jpeg = False
    else:
        options.jpeg = True

    extents = [float(t) for t in options.extents.split()]
    if len(extents) != 4:
        raise AttributeError('extents to be used for the cropped file.\n'
                             'needs to be a list or tuples of 4 floats\n'
                             "example:"
                             "--extents '-2362974.47956 -5097641.80634 2251415.52044 -1174811.80634'")
    options.extents = [str(s) for s in extents]

    do_work(input_file=options.input_file,
            mask_file=options.mask_file,
            output_file=options.output_file,
            resampling=options.resampling,
            extents=options.extents,
            jpeg=options.jpeg)

