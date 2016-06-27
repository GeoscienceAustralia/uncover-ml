"""
Extract patch features from a single geotiff.

.. program-output:: extractfeats --help
"""
import logging
from functools import partial
import os

import click as cl
import click_log as cl_log
import numpy as np
import pickle
from mpi4py import MPI


from uncoverml import patch
from uncoverml import stats
from uncoverml import geoio
import uncoverml.defaults as df

# logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def extract_transform(x, x_sets):
    # reshape
    x = x.reshape((x.shape[0], -1))
    if x_sets is not None:  # one-hot activated!
        x = stats.one_hot(x, x_sets)
    x = x.astype(float)
    return x


def unique(sets1, sets2, dtype):
    per_dim = zip(sets1, sets2)
    out_sets = [np.unique(np.concatenate(k, axis=0)) for k in per_dim]
    return out_sets

unique_op = MPI.Op.Create(unique, commute=True)


def compute_unique_values(x, comm):
    x_sets = None
    # check data is okay
    if x.dtype == np.dtype('float32') or x.dtype == np.dtype('float64'):
        log.warn("Cannot use one-hot for floating point data -- ignoring")
    else:
        local_sets = stats.sets(x)
        full_sets = comm.allreduce(local_sets, op=unique_op)
        total_dims = np.sum([len(k) for k in full_sets])
        log.info("Total features from one-hot encoding: {}".format(
            total_dims))
        if total_dims <= df.max_onehot_dims:
            x_sets = full_sets
        else:
            log.warn("Too many distinct values for one-hot encoding."
                     " If you're sure increase max value in default file.")
    return x_sets


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--patchsize', type=int,
           default=df.feature_patch_size, help="window width of patches, i.e. "
           "patchsize of 0 is a single pixel, patchsize of 1 is a 3x3 patch, "
           "etc")
@cl.option('--targets', type=cl.Path(exists=True), help="Optional hdf5 file "
           "for providing target points at which to evaluate feature. See "
           "maketargets for creating an appropriate target files.")
@cl.option('--onehot', is_flag=True, help="Produce a one-hot encoding for "
           "each channel in the data. Ignored for float-valued data. "
           "Uses -0.5 and 0.5)")
@cl.option('--settings', type=cl.Path(exists=True), help="file containing"
           " previous setting used for evaluating testing data. If provided "
           "all other option flags are ignored")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('name', type=str, required=True)
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
def main(name, geotiff, targets, onehot, patchsize, outputdir, settings):
    """
    Extract patch features from a single geotiff and output to HDF5 file chunks
    for distribution to worker nodes.

    Apart from extracting patches from images, this also has the following
    functionality,

    - chunk the original geotiff into many small HDF5 files for distributed
      work
    - One-hot encode intger-valued/categorical layers
    - Only extract patches at specified locations given a target file
    """

    # MPI globals
    comm = MPI.COMM_WORLD
    chunks = comm.Get_size()
    chunk_index = comm.Get_rank()

    # build full filename for geotiff
    full_filename = os.path.abspath(geotiff)
    log.info("Input file full path: {}".format(full_filename))

    # Print some helpful statistics about the full image
    full_image = geoio.Image(full_filename)

    total_dims = full_image.resolution[2]
    log.info("Image has resolution {}".format(full_image.resolution))
    log.info("Image has datatype {}".format(full_image.dtype))
    log.info("Image missing value: {}".format(full_image.nodata_value))

    # Compute the effective sampled resolution accounting for patchsize
    eff_shape = None
    eff_bbox = None
    if targets is None:
        start = [patchsize, patchsize]
        end_p1 = [full_image.xres - patchsize + 1,  # +1 because bbox
                  full_image.yres - patchsize + 1]  # +1 because bbox
        xy = np.array([start, end_p1])
        eff_bbox = full_image.pix2lonlat(xy)
        eff_shape = (full_image.xres - 2 * patchsize,
                     full_image.yres - 2 * patchsize)
        log.info("Effective input resolution "
                 "after patch extraction: {}".format(eff_shape))
        log.info("Effective bounding box after "
                 "patch extraction: {}".format(eff_bbox))

    # load settings
    f_args = {}
    if settings is not None:
        with open(settings, 'rb') as f:
            s = pickle.load(f)
            patchsize = s['cmd_args']['patchsize']
            log.info("Loading patchsize {} from settings file".format(
                patchsize))
            f_args.update(s['f_args'])

    image = geoio.Image(full_filename, chunk_index, chunks, patchsize)
    if targets is not None:
        #  we need full path for targets for the workers
        targets = os.path.abspath(targets)
        log.info("node {} reading target file {}".format(chunk_index, targets))
        x = patch.patches_at_target(image, patchsize, targets)
    else:
        x = patch.all_patches(image, patchsize)

    # compute settings
    if not settings:
        settings_dict = {}
        x_sets = compute_unique_values(x, comm) if onehot else None
        f_args['x_sets'] = x_sets
        settings_filename = os.path.join(outputdir, name + "_settings.bin")
        settings_dict["f_args"] = f_args
        settings_dict["cmd_args"] = {'patchsize': patchsize}
        if chunk_index == 0:
            log.info("Writing feature settings to {}".format(settings_filename))
            with open(settings_filename, 'wb') as f:
                pickle.dump(settings_dict, f)

    # We have all the information we need, now build the transform
    log.info("Constructing feature transformation function")
    f = partial(extract_transform, **f_args)

    outfile = geoio.output_filename(name, chunk_index,
                                    chunks, outputdir)
    if x is not None:
        log.info("Applying final transform and writing output files")
        f_x = f(x)
        total_dims = f_x.shape[1]
        write_ok = geoio.output_features(f_x, outfile, shape=eff_shape,
                                         bbox=eff_bbox)
    else:
        write_ok = geoio.output_blank(outfile)

    log.info("Output vector has length {}, dimensionality {}".format(
        full_image.resolution[0] * full_image.resolution[1], total_dims))

