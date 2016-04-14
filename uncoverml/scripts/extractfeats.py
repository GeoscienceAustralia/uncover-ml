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

def extract_transform(data, x_set, mean, sd):
    #Flatten patches
    data = data.reshape((data.shape[0], -1))
    if x_set is not None: # one-hot activated!
        data = parallel.one_hot(data, x_set)
    else:
        if mean is not None:
            data = parallel.centre(data, mean)
        if sd is not None:
            data = parallel.standardise(data, sd)
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
@cl.option('--centre', is_flag=True, help="Make data have mean zero")
@cl.option('--standardise', is_flag=True, help="Make all dimensions "
           "have unit variance")
@cl.option('--onehot', is_flag=True, help="Produce a one-hot encoding for "
           "each channel in the data. Only works for non-float values. "
           "Implies --centre and --standardise (ie uses -0.5 and 0.5)")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use", 
           default=None)
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
@cl.argument('name', type=str, required=True)
def main(geotiff, name, targets, centre, standardise, onehot, 
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
    log.info("Image has resolution {}".format(full_image.resolution))
    log.info("Image has datatype {}".format(full_image.dtype))

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
    # get number of points
    cluster.execute("x_count = parallel.node_count(x)")
    cluster.execute("x_full = parallel.node_full_count(x)")
    x_count = np.sum(np.array(cluster.pull('x_count')), axis=0)
    x_full = np.sum(np.array(cluster.pull('x_full')))
    
    fraction_missing =(1.0 - np.sum(x_count)/(x_full*x_count.shape[0]))*100.0
    log.info("Input data is {}% missing".format(fraction_missing))

    x_sets = None
    mean = None
    sd = None
    total_dims = x_count.shape[0]
    if onehot is True:
        #check data is okay
        dtype = image_dict[0].data().dtype
        if dtype == np.dtype('float32') or dtype == np.dtype('float64'):
            log.fatal("Cannot use one-hot for floating point data!")
            sys.exit(-1)
        cluster.execute("x_set = parallel.node_sets(x)")
        all_x_sets = cluster.pull('x_set')
        per_dim = zip(*all_x_sets)
        x_sets = [np.unique(np.concatenate(k,axis=0)) for k in per_dim]
        total_dims = np.sum([len(k) for k in x_sets])
        log.info("Total features from one-hot encoding: {}".format(
            total_dims))
        if total_dims > df.max_onehot_dims:
            log.fatal("Too many distinct values for one-hot encoding."
                      " If you're sure increase max value in default file.")
            sys.exit(-1)
    else:
        if centre is True:
            cluster.execute("x_sum = parallel.node_sum(x)")
            x_sum = np.sum(np.array(cluster.pull('x_sum')), axis=0)
            mean = x_sum / x_count
            log.info("Subtracting global mean {}".format(mean))
            cluster.push({"mean":mean})
            cluster.execute("x = parallel.centre(x, mean)")

        if standardise is True:
            cluster.execute("x_var = parallel.node_var(x)")
            x_var = np.sum(np.array(cluster.pull('x_var')),axis=0)
            sd = np.sqrt(x_var/x_count)
            log.info("Dividing through global standard deviation {}".format(sd))
            cluster.push({"sd":sd})
            cluster.execute("x = parallel.standardise(x, sd)")
    
    #We have all the information we need, now build the transform
    log.info("Constructing feature transformation function")
    f = partial(extract_transform, x_set=x_sets, mean=mean, sd=sd)

    log.info("Applying transform across nodes")
    # Apply the transformation function
    cluster.push({"f":f, "featurename":name, "outputdir":outputdir})
    cluster.execute("parallel.write_data(data_dict, f, featurename, outputdir)")

    log.info("Output vector has length {}, dimensionality {}".format(
        full_image.resolution[0] * full_image.resolution[1], total_dims))

    sys.exit(0)
