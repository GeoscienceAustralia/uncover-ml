"""
Compose multiple image features into a single feature vector.

.. program-output:: composefeats --help
"""

from __future__ import division

import logging
import sys
import os.path
import click as cl
import click_log as cl_log
import uncoverml.defaults as df
from uncoverml import geoio
from uncoverml import mpiops
from uncoverml import pipeline
from uncoverml import datatypes


log = logging.getLogger(__name__)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--config', type=cl.Path(exists=True), help="file containing"
           " previous setting used for evaluating testing data. If provided "
           "all other option flags are ignored")
@cl.option('--transform', type=str, help="The transform to apply to the data."
           "can be 'centre', 'standardise' or 'whiten'.")
@cl.option('--impute', is_flag=True, help="Impute any missing data")
@cl.option('--featurefraction', type=float, default=df.whiten_feature_fraction,
           help="The fraction of dimensions to keep for PCA transformed"
           " (whitened) data. Between 0 and 1. Only applies if --whiten given")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('name', type=str, required=True)
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(files, name, outputdir, transform,
         featurefraction, impute, config):
    """
    Compose multiple image features into a single feature vector.

    This tool also has the following functionality beyond combining image
    features (from, e.g. the output of extractfeats):

    - Impute missing features with the mean of the dimension
    - Centre the features (zero-mean)
    - Standardise the features (unit standard deviation)
    - Whiten the features (decorrelated all of the features and scale to have
        unit variance. This can be used to reduce the dimensionality of the
        features too).
    """
    # build full filenames
    hdf_infiles = [os.path.abspath(f) for f in files]
    files_ok = geoio.file_indices_okay(hdf_infiles)
    if not files_ok:
        sys.exit(-1)

    settings_infile = os.path.abspath(config) if config else None
    settings_outfile = os.path.join(outputdir, name + "_settings.bin")
    hdf_outfile = geoio.output_filename(name, mpiops.chunk_index,
                                        mpiops.chunks, outputdir)

    if settings_infile:
        settings = geoio.load_settings(settings_infile)
    else:
        settings = datatypes.ComposeSettings(impute=impute,
                                             transform=transform,
                                             featurefraction=featurefraction,
                                             impute_mean=None,
                                             mean=None,
                                             sd=None,
                                             eigvals=None,
                                             eigvecs=None)

    filename_dict = geoio.files_by_chunk(hdf_infiles)
    chunk_files = filename_dict[mpiops.chunk_index]
    x = geoio.load_and_cat(chunk_files)
    x_out, settings = pipeline.compose_features(x, settings)
    geoio.output_features(x, hdf_outfile)

    if not settings_infile:
        mpiops.run_once(geoio.save_settings, settings, settings_outfile)
