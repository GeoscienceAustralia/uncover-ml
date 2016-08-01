import os
from os.path import join, exists, basename
import glob
from optparse import OptionParser
from mpi4py import MPI
from utils.crop_mask_resample_reproject import (crop_reproject_resample,
                                                apply_mask)

def return_file_list(my_dir, extension):
    return glob.glob(join(my_dir, extension))


def convert_files(files, output_dir, mask_file, extents, resampling, jpeg):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    no_files = len(files)

    for i in range(rank, no_files, size):
        in_file = files[i]
        print("operating on {file} in process {rank}".format(file=in_file,
                                                             rank=rank))
        out_file = join(output_dir, basename(in_file))
        print('Crop/reproject/resample {file} using \n '
              'resampling: {resampling}\n'
              'output file: {out_file}'.format(file=in_file,
                                               resampling=resampling,
                                               out_file=out_file))
        crop_reproject_resample(input_file=in_file,
                                output_file=out_file,
                                sampling=resampling,
                                extents=extents)
        if mask_file:
            print('Masking {file} using\n '
                  'mask file: {mask_file} \n'
                  'output file: {out_file}'.format(file=in_file,
                                                   mask_file=mask_file,
                                                   out_file=out_file))
            apply_mask(mask_file=mask_file,
                       output_file=out_file,
                       extents=extents,
                       jpeg=jpeg)
    comm.Barrier()


if __name__ == '__main__':
    parser = OptionParser(usage='%prog -i input_dir -o output_dir'
                                ' -e extents\n'
                                ' -r resampling (optional)\n'
                                ' -m mask_file (optional)\n'
                                ' -j jpeg_file (optional)\n'
                                'Script')

    parser.add_option('-i', '--input', type=str, dest='input_dir',
                      help='name of input dir')

    parser.add_option('-o', '--out', type=str, dest='output_dir',
                      help='name of cropped output tif')

    parser.add_option('-m', '--mask', type=str, dest='mask_file',
                      help='name of mask file')

    parser.add_option('-e', '--extents', type=str, dest='extents',
                      help='extents to be used for the cropped file.\n'
                           'needs to be a list of 4 floats with spaces\n'
                           'example: '
                           "-e '150.91 -34.229999976 150.949166651 -34.17'")
    parser.add_option('-r', '--resampling', type=str, dest='sampling',
                      help='optional resampling algorithm to use')

    parser.add_option('-j', '--jpeg', type=int, dest='jpeg',
                      help='optional jpeg conversion. 0=no jpeg, 1=jpeg')

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

    if not options.sampling:  # if sampling is not given
        options.sampling = 'near'

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
                             "--extents '-2362974.47956 -5097641.80634 "
                             "2251415.52044 -1174811.80634'")

    options.extents = [str(s) for s in extents]
    files_to_convert = return_file_list(options.input_dir, '*.tif')
    convert_files(files_to_convert,
                  output_dir=options.output_dir,
                  mask_file=options.mask_file,
                  extents=options.extents,
                  resampling=options.sampling,
                  jpeg=options.jpeg
                  )
