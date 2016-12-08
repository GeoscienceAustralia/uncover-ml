"""
This script takes as input an input dir, an output dir, and optionally
a mask file, a resampling algorithm, and a reprojection and a jpeg flag.

It uses mpirun and the crop_mask_resample_reporject script functionality to
covert the tifs in the input dir, and saves the result in the output dir.

More details in crop_mask_resample_reproject.py
"""
import os
from os.path import join, exists, basename
import glob
from optparse import OptionParser
from mpi4py import MPI
import tempfile
import logging
from preprocessing.crop_mask_resample_reproject import \
    (do_work, TMPDIR, crop_reproject_resample)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def return_file_list(my_dir, extension):
    return glob.glob(join(my_dir, extension))


def convert_files(files, output_dir, opt, mask_file):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if mask_file:
        # temporary cropped mask file
        cropped_mask_file = tempfile.mktemp(suffix='.tif', dir=TMPDIR)

        # crop/reproject/resample the mask
        crop_reproject_resample(opt.mask_file,
                                cropped_mask_file,
                                sampling='near',
                                extents=opt.extents,
                                reproject=opt.reproject)
    else:
        cropped_mask_file = opt.mask_file

    for i in range(rank, len(files), size):
        in_file = files[i]
        print('============file no: {} of {}==========='.format(i, len(files)))
        log.info("operating on {file} in process {rank}".format(file=in_file,
                                                                rank=rank))
        out_file = join(output_dir, basename(in_file))
        log.info('Output file: {}'.format(out_file))
        do_work(input_file=in_file,
                mask_file=cropped_mask_file,
                output_file=out_file,
                options=options)

    if mask_file:
        os.remove(cropped_mask_file)
        log.info('removed intermediate cropped '
                 'mask file {}'.format(cropped_mask_file))

    comm.Barrier()


# TODO: use click here
if __name__ == '__main__':
    parser = OptionParser(usage='%prog -i input_dir -o output_dir'
                                ' -e extents (optional)\n'
                                ' -r resampling (optional)\n'
                                ' -m mask_file (optional)\n'
                                ' -j jpeg_file (optional)\n'
                                ' -p reproject (optional)\n'
                                'Crop, resample, and '
                                'optionally mask and reproject a larger '
                                'interferrogram into smaller ones.'
                                'Optionally, create outout jpegs')

    parser.add_option('-i', '--input', type=str, dest='input_dir',
                      help='name of input dir or tif')

    parser.add_option('-o', '--out', type=str, dest='output_dir',
                      help='name of output dir or cropped/reprojected tif')

    parser.add_option('-m', '--mask', type=str, dest='mask_file',
                      help='name of mask file')

    parser.add_option('-e', '--extents', type=str, dest='extents',
                      help='extents (in target coordinates)'
                           'to be used for the cropped file.\n'
                           'needs to be a list of 4 floats with spaces\n'
                           'example: '
                           "-e '150.91 -34.229999976 150.949166651 -34.17'. \n"
                           "If no extents is provided, use whole image")

    parser.add_option('-r', '--resampling', type=str, dest='resampling',
                      help='optional resampling algorithm to use')

    parser.add_option('-j', '--jpeg', type=int, dest='jpeg',
                      help='optional jpeg conversion. 0=no jpeg, 1=jpeg')

    parser.add_option('-p', '--reproject', type=int, dest='reproject',
                      help='optional reprojection to Lambart Conic. ')

    options, args = parser.parse_args()

    if not options.input_dir:  # if filename is not given
        parser.error('Input dir name not given.')

    if not exists(options.input_dir):
        raise IOError(msg='Dir {} does not exist'.format(options.input_dir))

    if not options.output_dir:  # if filename is not given
        parser.error('Output dir name not given.')

    # create output dir if it does not exist
    if not exists(options.output_dir):
        os.mkdir(options.output_dir)

    if not options.resampling:  # if sampling is not given
        options.resampling = 'near'

    if not options.jpeg:  # if jpeg is not given
        options.jpeg = False
    else:
        options.jpeg = True

    if not options.reproject:  # if reproject is not supplied
        options.reproject = False
    else:
        options.reproject = True

    if options.extents:
        extents = [float(t) for t in options.extents.split()]
        if len(extents) != 4:
            raise AttributeError('extents to be used for the cropped file.\n'
                                 'needs to be a list or tuples of 4 floats\n'
                                 "example:"
                                 "--extents '-2362974.47956 -5097641.80634 "
                                 "2251415.52044 -1174811.80634'")
        options.extents = [str(s) for s in extents]
    else:
        logging.info('No extents specified. Using whole image for projection')
        options.extents = None

    if os.path.isdir(options.input_dir):
        files_to_convert = return_file_list(options.input_dir, '*.tif')
    else:
        files_to_convert = [options.input_dir]

    convert_files(files_to_convert,
                  output_dir=options.output_dir,
                  opt=options,
                  mask_file=options.mask_file,
                  )
