"""
Compose multiple image features into a single feature vector.

.. program-output:: composefeats --help
"""
import logging
import sys
import os.path
import click as cl
import numpy as np
import json
import time
import uncoverml.defaults as df
from functools import partial
from uncoverml import parallel
from uncoverml import geoio
from uncoverml import feature

log = logging.getLogger(__name__)

def transform(x, eigvecs, eigvals, fraction):
    
    if eigvecs is not None:
        ndims = x.shape[1]
        #make sure 1 <= keepdims <= ndims
        keepdims = min(max(1,int(ndims * fraction)), ndims)
        mat = eigvecs[:, -keepdims:]
        vec = eigvals[-keepdims:]
        x_t = np.ma.dot(x, mat) / np.sqrt(vec)
    else:
        x_t = x
    return x_t


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", 
           default=df.quiet_logging)
@cl.option('--whiten', is_flag=True, help="Data is a unit ball")
@cl.option('--featurefraction', type=float, default=df.whiten_feature_fraction,
           help="The fraction of dimensions to keep for PCA transformed"
           " (whitened) data. Between 0 and 1. Only applies if --whiten given")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use", 
           default=None)
@cl.argument('featurename', type=str, required=True) 
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(files, featurename, quiet, outputdir, ipyprofile,
         whiten, featurefraction):
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
    
    # Define the transform function to build the features
    cluster = parallel.direct_view(ipyprofile, nchunks)

    #Local Mode!
    # data_dict = feature.load_data(filename_dict, range(nchunks))
    # x = feature.data_vector(data_dict)
    # x_n = parallel.node_count(x)
    # x_outer = parallel.node_outer(x)
    # cov = x_outer/x_n
    # eigvals, eigvecs = np.linalg.eigh(cov)
    # f = partial(transform, eigs=eigvecs, fraction=featurefraction)
    # parallel.write_data(data_dict, f, "localfeature", outputdir)

    # Load the data into a dict on each client
    # Note chunk_indices is a global with different value on each node
    cluster.push({"filename_dict":filename_dict})
    cluster.execute("data_dict = feature.load_data(filename_dict, chunk_indices)")
    cluster.execute("x = feature.data_vector(data_dict)")

    # Get count data
    cluster.execute("x_n = parallel.node_count(x)")
    cluster.execute("x_full = parallel.node_full_count(x)")
    x_n = np.sum(np.array(cluster.pull('x_n'),dtype=float), axis=0)
    x_full = np.sum(np.array(cluster.pull('x_full')))
    log.info("Total input dimensionality: {}".format(x_n.shape[0]))
    fraction_missing =(1.0 - np.sum(x_n)/(x_full*x_n.shape[0]))*100.0
    log.info("Input data is {}% missing".format(fraction_missing))
    
    eigvecs = None
    eigvals = None
    if whiten is True:
        cluster.execute("x_outer = parallel.node_outer(x)")
        outer = np.sum(np.array(cluster.pull('x_outer')),axis=0)
        cov = outer/x_n
        eigvals, eigvecs = np.linalg.eigh(cov)
        log.info("Whitening and keeping {} dimensions".format(
            int(x_n.shape[0]*featurefraction)))

    #We have all the information we need, now build the transform
    f = partial(transform, eigvecs=eigvecs, eigvals=eigvals,
                fraction=featurefraction)

    # Apply the transformation function
    cluster.push({"f":f, "featurename":featurename, "outputdir":outputdir})
    cluster.execute("parallel.write_data(data_dict, f, featurename, outputdir)")
    sys.exit(0)

