"""
Compose multiple image features into a single feature vector.

.. program-output:: composefeats --help
"""

from __future__ import division

import logging
import sys
import os.path
import click as cl
import numpy as np
import uncoverml.defaults as df
from functools import partial
from uncoverml import parallel
from uncoverml import geoio
from uncoverml import feature
import pickle

log = logging.getLogger(__name__)


def transform(x, impute_mean, mean, sd, eigvecs, eigvals, featurefraction):

    if impute_mean is not None:
        x = parallel.impute_with_mean(x, impute_mean)
    if mean is not None:
        x = parallel.centre(x, mean)
    if sd is not None:
        x = parallel.standardise(x, sd)
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
    # Get count data
    cluster.execute("x = feature.data_vector(data_dict)")
    cluster.execute("x_n = parallel.node_count(x)")
    cluster.execute("x_full = parallel.node_full_count(x)")
    x_n = np.sum(np.array(cluster.pull('x_n'), dtype=float), axis=0)
    x_full = np.sum(np.array(cluster.pull('x_full')))
    out_dims = x_n.shape[0]
    log.info("Total input dimensionality: {}".format(x_n.shape[0]))
    fraction_missing = (1.0 - np.sum(x_n)/(x_full*x_n.shape[0]))*100.0
    log.info("Input data is {}% missing".format(fraction_missing))

    impute_mean = None
    if impute is True:
        cluster.execute("impute_sum = parallel.node_sum(x)")
        impute_sum = np.sum(np.array(cluster.pull('impute_sum')), axis=0)
        impute_mean = impute_sum / x_n
        log.info("Imputing missing data from mean {}".format(impute_mean))
        cluster.push({'impute_mean': impute_mean})
        cluster.execute("x = parallel.impute_with_mean(x, impute_mean)")
        cluster.execute("x_n = parallel.node_count(x)")
        x_n = np.sum(np.array(cluster.pull('x_n'), dtype=float), axis=0)

    mean = None
    if centre is True:
        cluster.execute("x_sum = parallel.node_sum(x)")
        x_sum = np.sum(np.array(cluster.pull('x_sum')), axis=0)
        mean = x_sum / x_n
        log.info("Subtracting global mean {}".format(mean))
        cluster.push({"mean": mean})
        cluster.execute("x = parallel.centre(x, mean)")

    sd = None
    if standardise is True:
        cluster.execute("x_var = parallel.node_var(x)")
        x_var = np.sum(np.array(cluster.pull('x_var')), axis=0)
        sd = np.sqrt(x_var/x_n)
        log.info("Dividing through global standard deviation {}".format(sd))
        cluster.push({"sd": sd})
        cluster.execute("x = parallel.standardise(x, sd)")

    eigvecs = None
    eigvals = None
    if whiten is True:
        cluster.execute("x_outer = parallel.node_outer(x)")
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
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
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
def main(files, featurename, quiet, outputdir, ipyprofile,
         centre, standardise, whiten, featurefraction, impute, settings):
    """ TODO
    """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # build full filenames
    full_filenames = [os.path.abspath(f) for f in files]
    log.debug("Input files: {}".format(full_filenames))

    # verify the files are all present
    files_ok = geoio.file_indices_okay(full_filenames)
    if not files_ok:
        sys.exit(-1)

    # build the images
    filename_dict = geoio.files_by_chunk(full_filenames)
    nchunks = len(filename_dict)

    # Get attribs if they exist
    eff_shape, eff_bbox = feature.load_attributes(filename_dict)

    # Define the transform function to build the features
    cluster = parallel.direct_view(ipyprofile, nchunks)

    # load settings
    f_args = {}
    if settings is not None:
        with open(settings, 'rb') as f:
            s = pickle.load(f)
            f_args.update(s['f_args'])

    # Load the data into a dict on each client
    # Note chunk_indices is a global with different value on each node
    cluster.push({"filename_dict": filename_dict})
    cluster.execute("data_dict = feature.load_data("
                    "filename_dict, chunk_indices)")

    if settings is None:
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

    # Apply the transformation function
    cluster.push({"f": f, "featurename": featurename, "outputdir": outputdir,
                  "shape": eff_shape, "bbox": eff_bbox})
    log.info("Applying final transform and writing output files")
    cluster.execute("parallel.write_data(data_dict, f, "
                    "featurename, outputdir, shape, bbox)")
    sys.exit(0)
