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

def whiten(x, mean, mat):
    x_0 = x - mean
    x_w = np.dot(x_0, mat)
    return x_w

@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", 
           default=df.quiet_logging)
@cl.option('--standalone', is_flag=True, default=df.standalone)
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use", 
           default=None)
@cl.argument('featurename', type=str, required=True) 
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(files, featurename, standalone, quiet, outputdir, ipyprofile):
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
    cluster = parallel.direct_view(ipyprofile, nchunks) \
        if not standalone else None

    cluster.clear()
    
    # Load the data
    cluster.apply(parallel.load_data, filename_chunks)

    #get the mean and cov
    sums_and_counts = parallel.map_over_data(parallel.mean, cluster)
    sums, ns = zip(*sums_and_counts)
    full_sum = np.sum(np.array(sums),axis=0)
    n = np.sum(np.array(ns), dtype=float)
    mean = full_sum / n
    f_demean = partial(parallel.cov, mean=mean)
    outers = parallel.map_over_data(f_demean, cluster) 
    full_outer = np.sum(np.array(outers),axis=0)
    cov = full_outer / n
    # whitening transform
    w, v = np.linalg.eigh(cov)
    ndims = 2
    mat = v[:,-ndims:]
        
    # build the whitening transform
    transform = partial(whiten, mean=mean, mat=mat)

    # Apply the transformation function
    cluster.apply(parallel.write_data, transform, featurename, 
                              outputdir)
    sys.exit(0)

