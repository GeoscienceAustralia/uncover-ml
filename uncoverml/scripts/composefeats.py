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
from uncoverml import geoio
from uncoverml import stats
import pickle


from mpi4py import MPI

log = logging.getLogger(__name__)


def transform_with_params(x, impute_mean, mean, sd, eigvecs,
                          eigvals, featurefraction, comm):

    if impute_mean is not None:
        stats.impute_with_mean(x, impute_mean)

    if mean is not None:

        # Always subtract mean before whiten
        if sd is not None:
            stats.standardise(x.data, sd, mean)
        else:
            stats.centre(x.data, mean)

        if eigvecs is not None:
            ndims = x.shape[1]
            # make sure 1 <= keepdims <= ndims
            keepdims = min(max(1, int(ndims * featurefraction)), ndims)
            mat = eigvecs[:, -keepdims:]
            vec = eigvals[-keepdims:]
            x = np.ma.dot(x, mat, strict=True) / np.sqrt(vec)

    return x


def sum_axis_0(x, y, dtype):
    s = np.sum(np.array([x, y]), axis=0)
    return s

sum0_op = MPI.Op.Create(sum_axis_0, commute=True)


def transform(x, impute, centre, standardise, whiten,
              featurefraction, comm):

    x_n_local = stats.count(x)
    x_n = comm.allreduce(x_n_local, op=MPI.SUM)
    x_full_local = stats.full_count(x)
    x_full = comm.allreduce(x_full_local, op=MPI.SUM)

    out_dims = x_n.shape[0]
    log.info("Total input dimensionality: {}".format(x_n.shape[0]))
    fraction_missing = (1.0 - np.sum(x_n) / (x_full * x_n.shape[0])) * 100.0
    log.info("Input data is {}% missing".format(fraction_missing))

    impute_mean = None
    if impute is True:
        local_impute_sum = stats.sum(x)
        impute_sum = comm.allreduce(local_impute_sum, op=sum0_op)
        impute_mean = impute_sum / x_n
        log.info("Imputing missing data from mean {}".format(impute_mean))
        stats.impute_with_mean(x, impute_mean)
        x_n_local = stats.count(x)
        x_n = comm.allreduce(x_n_local, op=MPI.SUM)

    out_mean = None
    if centre or standardise or whiten:
        x_sum_local = stats.sum(x)
        x_sum = comm.allreduce(x_sum_local, op=sum0_op)
        mean = x_sum / x_n
        out_mean = mean.copy()

    if centre is True and not standardise:
        log.info("Subtracting global mean {}".format(mean))
        stats.centre(x.data, mean)
        mean = np.zeros_like(mean)

    sd = None
    if standardise is True:
        x_var_local = stats.var(x, mean)
        x_var = comm.allreduce(x_var_local, op=sum0_op)
        sd = np.sqrt(x_var / x_n)
        log.info("Dividing through global standard deviation {}".format(sd))
        stats.standardise(x.data, sd, mean)
        mean = np.zeros_like(mean)

    eigvecs = None
    eigvals = None
    if whiten is True:
        x_outer_local = stats.outer(x, mean)
        outer = comm.allreduce(x_outer_local, op=sum0_op)
        cov = outer / x_n
        eigvals, eigvecs = np.linalg.eigh(cov)
        out_dims = int(out_dims * featurefraction)
        log.info("Whitening and keeping {} dimensions".format(out_dims))
        ndims = x.shape[1]
        # make sure 1 <= keepdims <= ndims
        keepdims = min(max(1, int(ndims * featurefraction)), ndims)
        mat = eigvecs[:, -keepdims:]
        vec = eigvals[-keepdims:]
        x = np.ma.dot(x, mat, strict=True) / np.sqrt(vec)

    d = {"impute_mean": impute_mean, "mean": out_mean, "sd": sd,
         "eigvecs": eigvecs, "eigvals": eigvals,
         "featurefraction": featurefraction}

    return x, d


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
@cl.argument('featurename', type=str, required=True)
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(files, featurename, outputdir, centre, standardise,
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
    # MPI globals
    comm = MPI.COMM_WORLD
    chunks = comm.Get_size()
    chunk_index = comm.Get_rank()

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
    chunk_files = filename_dict[chunk_index]
    log.info("Node {} loading {}".format(chunk_index, chunk_files))
    x = geoio.load_and_cat(filename_dict[chunk_index])

    # not everyone has data
    has_data = x is not None
    has_data_mask = comm.allgather(has_data)
    key = chunk_index if has_data else -1 * chunk_index
    dcomm_root = has_data_mask.index(True)
    dcomm = comm.Split(has_data, key)


    # load settings
    f_args = {}
    if settings is not None:
        with open(settings, 'rb') as f:
            s = pickle.load(f)
            f_args.update(s['f_args'])
        if has_data:
            f = partial(transform_with_params, **f_args)
            xt = f(x, comm=dcomm)
    else:
        if has_data:
            xt, params = transform(x, impute, centre, standardise, whiten,
                                   featurefraction, dcomm)

            f_args.update(params)
            settings_dict = {}
            settings_dict["f_args"] = f_args
            # write settings
            if chunk_index == dcomm_root:
                settings_filename = os.path.join(outputdir,
                                                 featurename + "_settings.bin")
                log.info("Writing feature settings to {}".format(
                    settings_filename))
                with open(settings_filename, 'wb') as f:
                    pickle.dump(settings_dict, f)

    # We have all the information we need, now build the transform
    outfile = geoio.output_filename(featurename, chunk_index,
                                    chunks, outputdir)
    if has_data:
        log.info("Applying final transform and writing output files")
        write_ok = geoio.output_features(xt, outfile, shape=eff_shape,
                                         bbox=eff_bbox)
    else:
        write_ok = geoio.output_blank(outfile)
