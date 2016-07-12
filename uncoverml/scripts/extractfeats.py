"""
Extract patch features from a single geotiff.

.. program-output:: extractfeats --help
"""
import logging
import os

import click as cl
import click_log as cl_log
import pickle

import uncoverml.defaults as df
from uncoverml import mpiops
from uncoverml import patch
from uncoverml import stats
from uncoverml import geoio

# logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class PickledSettings:
    def from_file(settings_file):
        s = pickle.load(settings_file)
        return s

    def save(self, settings_file):
        with open(settings_file, 'wb') as f:
            pickle.dump(self, f)


class ExtractSettings(PickledSettings):

    def __init__(self, onehot, x_sets, patchsize):
        self.onehot = onehot
        self.x_sets = x_sets
        self.patchsize = patchsize


def transform(x, x_sets):
    x = x.reshape(x.shape[0], -1)
    if x_sets:
        x = stats.one_hot(x, x_sets)
    x = x.astype(float)
    return x


def extract_features(settings, target_infile, geotiff_infile, hdf_outfile):

    # Compute the effective sampled resolution accounting for patchsize
    full_image = geoio.Image(geotiff_infile)
    eff_shape = full_image.patched_shape(settings.patchsize)
    eff_bbox = full_image.patched_bbox(settings.patchsize)

    image = geoio.Image(geotiff_infile, mpiops.chunk_index,
                        mpiops.chunks, settings.patchsize)

    x = patch.load(image, settings.patchsize, target_infile)

    if settings.onehot and not settings.x_sets:
        settings.x_sets = mpiops.compute_unique_values(x, df.max_onehot_dims)

    if x is not None:
        x = transform(x, settings.x_sets)
        geoio.output_features(x, hdf_outfile, shape=eff_shape, bbox=eff_bbox)
    else:
        geoio.output_blank(hdf_outfile, shape=eff_shape, bbox=eff_bbox)

    return settings


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
@cl.option('--config', type=cl.Path(exists=True), help="file containing"
           " previous setting used for evaluating testing data. If provided "
           "all other option flags are ignored")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('name', type=str, required=True)
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
def main(name, geotiff, targets, onehot, patchsize, outputdir, config):
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

    # Full paths
    target_infile = os.path.abspath(targets) if targets else None
    geotiff_infile = os.path.abspath(geotiff)
    settings_infile = os.path.abspath(config) if config else None
    settings_outfile = os.path.join(outputdir, name + "_settings.bin")
    hdf_outfile = geoio.output_filename(name, mpiops.chunk_index,
                                        mpiops.chunks, outputdir)

    if settings_infile:
        settings = ExtractSettings.from_file(settings_infile)
    else:
        settings = ExtractSettings(onehot=onehot, x_sets=None,
                                   patchsize=patchsize)

    settings = extract_features(settings, target_infile,
                                geotiff_infile, hdf_outfile)

    if not settings_infile:
        mpiops.run_once(settings.save, settings_outfile)
