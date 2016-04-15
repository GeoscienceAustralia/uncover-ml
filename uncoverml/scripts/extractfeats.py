"""
Extract patch features from a single geotiff.

.. program-output:: extractfeats --help
"""
import logging
from functools import partial
import os
import click as cl
import uncoverml.feature as feat
from uncoverml import geoio
import uncoverml.defaults as df
from uncoverml import parallel
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
import sys
from uncoverml import feature

log = logging.getLogger(__name__)

def extract_transform(data, x_set):
    #Flatten patches
    data = data.reshape((data.shape[0], -1))
    if x_set is not None: # one-hot activated!
        data = parallel.one_hot(data, x_set)
    data = data.astype(float)
    return data
        


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--patchsize', type=int,
           default=df.feature_patch_size, help="window size of patches")
@cl.option('--chunks', type=int, default=df.work_chunks, help="Number of "
           "chunks in which to split the computation and output")
@cl.option('--targets', type=cl.Path(exists=True), help="Optional hdf5 file "
           "for providing target points at which to evaluate feature. See "
           "maketargets for creating an appropriate target files.")
@cl.option('--onehot', is_flag=True, help="Produce a one-hot encoding for "
           "each channel in the data. Ignored for float-valued data. "
           "Uses -0.5 and 0.5)")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use", 
           default=None)
@cl.argument('name', type=str, required=True)
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
def main(name, geotiff, targets, onehot, 
         chunks, patchsize, quiet, outputdir, ipyprofile):
    """ TODO
    """
    
    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # build full filename for geotiff
    full_filename = os.path.abspath(geotiff)
    log.info("Input file full path: {}".format(full_filename))

    # Print some helpful statistics about the full image
    full_image = geoio.Image(full_filename)
    total_dims = full_image.resolution[2]
    log.info("Image has resolution {}".format(full_image.resolution))
    log.info("Image has datatype {}".format(full_image.dtype))
    log.info("Image missing value: {}".format(full_image.nodata_value))

    # build the chunk->image dictionary for the input data
    image_dict = {i:geoio.Image(full_filename, i, chunks, patchsize) 
                    for i in range(chunks)}

    # Initialise the cluster
    cluster = parallel.direct_view(ipyprofile, chunks)

    # Load the data into a dict on each client
    # Note chunk_indices is a global with different value on each node
    cluster.push({"image_dict":image_dict, "patchsize":patchsize,
                  "targets":targets})
    cluster.execute("data_dict = feature.load_image_data( "
                    "image_dict, chunk_indices, patchsize, targets)")
    cluster.execute("x = feature.image_data_vector(data_dict)")

    x_sets = None
    if onehot is True:
        #check data is okay
        dtype = full_image.dtype
        if dtype == np.dtype('float32') or dtype == np.dtype('float64'):
            log.warn("Cannot use one-hot for floating point data -- ignoring")
            onehot = False
        else:
            cluster.execute("x_set = parallel.node_sets(x)")
            all_x_sets = cluster.pull('x_set')
            per_dim = zip(*all_x_sets)
            potential_x_sets = [np.unique(np.concatenate(k,axis=0)) for k in per_dim]
            total_dims = np.sum([len(k) for k in potential_x_sets])
            log.info("Total features from one-hot encoding: {}".format(
                total_dims))
            if total_dims > df.max_onehot_dims:
                log.warn("Too many distinct values for one-hot encoding."
                          " If you're sure increase max value in default file.")
                onehot = False
            else:
                # We'll actually do the one-hot now
                log.info("One-hot encoding data")
                x_sets = potential_x_sets
                cluster.push({"x_sets":x_sets})
                cluster.execute("x = parallel.one_hot(x, x_sets)")
                log.info("Data successfully one-hot encoded")

    
    #We have all the information we need, now build the transform
    log.info("Constructing feature transformation function")
    f = partial(extract_transform, x_set=x_sets)

    log.info("Applying transform across nodes")
    # Apply the transformation function
    cluster.push({"f":f, "featurename":name, "outputdir":outputdir})
    log.info("Applying final transform and writing output files")
    cluster.execute("parallel.write_data(data_dict, f, featurename, outputdir)")

    log.info("Output vector has length {}, dimensionality {}".format(
        full_image.resolution[0] * full_image.resolution[1], total_dims))

    sys.exit(0)
