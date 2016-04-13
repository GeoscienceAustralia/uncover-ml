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
from uncoverml import feature as feat

log = logging.getLogger(__name__)

def transform(x, eigs, fraction):
    
    if eigs is not None:
        ndims = x.shape[1]
        #make sure 1 <= keepdims <= ndims
        keepdims = min(max(1,int(ndims * fraction)), ndims)
        mat = eigs[:, -keepdims:]
        x_t = np.dot(x, mat)
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
    filename_chunks = geoio.files_by_chunk(full_filenames)
    nchunks = len(filename_chunks)
    
    # Define the transform function to build the features
    cluster = parallel.direct_view(ipyprofile, nchunks)

    # Load the data into a dict on each client
    # Note chunk_indices is a global with different value on each node
    cluster.push({"chunk_dict":filename_chunks})
    cluster.execute("data = parallel.load_and_cat(chunk_indices, chunk_dict)")
    
    #create a simple per-node vector for easy statistics
    cluster.execute("x = parallel.merge_clusters(data, chunk_indices)")
    cluster.execute("x_n = parallel.node_count(x)")
    x_n = np.sum(np.array(cluster.pull('x_n'),dtype=float))

    eigvecs = None
    if whiten is True:
        cluster.execute("x_outer = parallel.node_outer(x)")
        outer = np.sum(np.array(cluster.pull('x_outer')),axis=0)
        cov = outer/x_n
        eigvals, eigvecs = np.linalg.eigh(cov)

    #We have all the information we need, now build the transform
    f = partial(transform, eigs=eigvecs, fraction=featurefraction)

    # Apply the transformation function
    cluster.push({"f":f, "featurename":featurename, "outputdir":outputdir})
    cluster.execute("parallel.write_data(data, f, featurename, outputdir)")
    sys.exit(0)

