"""
Extract patch features from a single geotiff.

.. program-output:: extractfeats --help
"""
import logging
import os

import click as cl
import click_log as cl_log

import uncoverml.defaults as df
from uncoverml import mpiops
from uncoverml import pipeline
from uncoverml import geoio

log = logging.getLogger(__name__)


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
        settings = pipeline.ExtractSettings.from_file(settings_infile)
    else:
        settings = pipeline.ExtractSettings(onehot=onehot, x_sets=None,
                                            patchsize=patchsize)

    settings = pipeline.extract_features(settings, target_infile,
                                         geotiff_infile, hdf_outfile)

    if not settings_infile:
        mpiops.run_once(settings.save, settings_outfile)
