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
from uncoverml import parallel
from uncoverml import geoio
from uncoverml import feature as feat

log = logging.getLogger(__name__)

def transform(array_list):
    return np.concatenate(array_list, axis=1)

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
    
    # Load the data
    cluster.apply(parallel.load_data, filename_chunks)

    # Apply the transformation function
    print("writing output...")
    filenames = cluster.apply(parallel.write_data, transform, featurename, 
                              outputdir)
    print(filenames)
    print('Complete')

    sys.exit(0)

