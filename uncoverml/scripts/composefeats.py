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
import numpy as np
import uncoverml.defaults as df
from functools import partial
from uncoverml import parallel
from uncoverml import geoio
from uncoverml import stats
import pickle


log = logging.getLogger(__name__)


def transform(x, impute_mean, mean, sd, eigvecs, eigvals, featurefraction):

    if impute_mean is not None:
        x = stats.impute_with_mean(x, impute_mean)
    if mean is not None:
        x = stats.centre(x, mean)
    if sd is not None:
        x = stats.standardise(x, sd)
    if eigvecs is not None:
        ndims = x.shape[1]
        # make sure 1 <= keepdims <= ndims
        keepdims = min(max(1, int(ndims * featurefraction)), ndims)
        mat = eigvecs[:, -keepdims:]
        vec = eigvals[-keepdims:]
        x = np.ma.dot(x, mat, strict=True) / np.sqrt(vec)

    return x


def compute_statistics(impute, centre, standardise, whiten,
                       featurefraction, cluster):
    # copy data so as not to screw with original
    cluster.execute("xs = np.ma.copy(x)")
    # Get count data
    cluster.execute("x_n = stats.count(xs)")
    cluster.execute("x_full = stats.full_count(xs)")
    x_n = np.sum(np.array(cluster.pull('x_n'), dtype=float), axis=0)
    x_full = np.sum(np.array(cluster.pull('x_full')))
    out_dims = x_n.shape[0]
    log.info("Total input dimensionality: {}".format(x_n.shape[0]))
    fraction_missing = (1.0 - np.sum(x_n) / (x_full*x_n.shape[0]))*100.0
    log.info("Input data is {}% missing".format(fraction_missing))

    impute_mean = None
    if impute is True:
        cluster.execute("impute_sum = stats.sum(xs)")
        impute_sum = np.sum(np.array(cluster.pull('impute_sum')), axis=0)
        impute_mean = impute_sum / x_n
        log.info("Imputing missing data from mean {}".format(impute_mean))
        cluster.push({'impute_mean': impute_mean})
        cluster.execute("xs = stats.impute_with_mean(xs, impute_mean)")
        cluster.execute("x_n = stats.count(xs)")
        x_n = np.sum(np.array(cluster.pull('x_n'), dtype=float), axis=0)

    mean = None
    if centre is True:
        cluster.execute("x_sum = stats.sum(xs)")
        x_sum = np.sum(np.array(cluster.pull('x_sum')), axis=0)
        mean = x_sum / x_n
        log.info("Subtracting global mean {}".format(mean))
        cluster.push({"mean": mean})
        cluster.execute("xs = stats.centre(xs, mean)")

    sd = None
    if standardise is True:
        cluster.execute("x_var = stats.var(xs)")
        x_var = np.sum(np.array(cluster.pull('x_var')), axis=0)
        sd = np.sqrt(x_var/x_n)
        log.info("Dividing through global standard deviation {}".format(sd))
        cluster.push({"sd": sd})
        cluster.execute("xs = stats.standardise(xs, sd)")

    eigvecs = None
    eigvals = None
    if whiten is True:
        cluster.execute("x_outer = stats.outer(xs)")
        outer = np.sum(np.array(cluster.pull('x_outer')), axis=0)
        cov = outer/x_n
        eigvals, eigvecs = np.linalg.eigh(cov)
        out_dims = int(out_dims*featurefraction)
        log.info("Whitening and keeping {} dimensions".format(out_dims))

    d = {"impute_mean": impute_mean, "mean": mean, "sd": sd,
         "eigvecs": eigvecs, "eigvals": eigvals,
         "featurefraction": featurefraction}
    return d


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--settings', type=cl.Path(exists=True), help="file containing"
           " previous setting used for evaluating testing data. If provided "
           "all other option flags are ignored")
@cl.option('--centre', is_flag=True, help="Make data have mean zero")
@cl.option('--standardise', is_flag=True, help="Make all dimensions "
           "have unit variance")
@cl.option('--whiten', is_flag=True, help="Data is a unit ball")
@cl.option('--impute', is_flag=True, help="Impute any missing data")
@cl.option('--featurefraction', type=float, default=df.whiten_feature_fraction,
           help="The fraction of dimensions to keep for PCA transformed"
           " (whitened) data. Between 0 and 1. Only applies if --whiten given")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use",
           default=None)
@cl.argument('featurename', type=str, required=True)
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(files, featurename, outputdir, ipyprofile, centre, standardise,
         whiten, featurefraction, impute, settings):
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
    full_filenames = [os.path.abspath(f) for f in files]
    log.debug("Input files: {}".format(full_filenames))

    # verify the files are all present
    files_ok = geoio.file_indices_okay(full_filenames)
    if not files_ok:
        sys.exit(-1)

    # build the images
    filename_dict = geoio.files_by_chunk(full_filenames)

    # Get attribs if they exist
    eff_shape, eff_bbox = geoio.load_attributes(filename_dict)

    # Define the transform function to build the features
    eff_nchunks = len(filename_dict)
    cluster = parallel.direct_view(ipyprofile, eff_nchunks)

    # Load the data into a dict on each client
    # Note chunk_index is a global with different value on each node
    for i in range(len(cluster)):
        cluster.push({"filenames": filename_dict[i]}, targets=i)
    cluster.execute("x = geoio.load_and_cat(filenames)")

    # load settings
    f_args = {}
    if settings is not None:
        with open(settings, 'rb') as f:
            s = pickle.load(f)
            f_args.update(s['f_args'])
    else:
        f_args.update(compute_statistics(impute, centre, standardise, whiten,
                                         featurefraction, cluster))
        settings_dict = {}
        settings_filename = os.path.join(outputdir,
                                         featurename + "_settings.bin")
        settings_dict["f_args"] = f_args
        log.info("Writing feature settings to {}".format(settings_filename))
        with open(settings_filename, 'wb') as f:
            pickle.dump(settings_dict, f)

    # We have all the information we need, now build the transform
    f = partial(transform, **f_args)

    parallel.apply_and_write(cluster, f, "x", featurename, outputdir,
                             eff_shape, eff_bbox)

    # Make sure client cleans up
    cluster.client.purge_everything()
    cluster.client.close()
