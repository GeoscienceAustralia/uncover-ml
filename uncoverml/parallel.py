import numpy as np
import os.path
import time
import click
import ipyparallel as ipp
from uncoverml import feature
import logging
import signal

log = logging.getLogger(__name__)

def direct_view(profile, nchunks):
    client = ipp.Client(profile=profile) if profile is not None else ipp.Client()
    c = client[:] # direct view

    # Initialise the cluster
    c.block = True
    # Ensure this module's requirments are imported externally
    c.execute('from uncoverml import feature')
    c.execute('from uncoverml import parallel')
    c.execute('import numpy as np')
    
    # Assign the chunks to workers
    nworkers = len(c)
    log.info("dividing {} work chunks between {} engines".format(
        nchunks, nworkers))
    for i, indices in enumerate(np.array_split(np.arange(nchunks),nworkers)):
        cmd = "chunk_indices = {}".format(indices.tolist())
        log.debug("assigning engine {} chunks {}".format(i, indices))
        c.execute(cmd, targets=i)
    return c

# def load_and_cat(chunk_indices, chunk_dict):
#     """
#     loads the data and concatenates it by dimension

#     Parameters 
#     ==========
#         chunk_indices: the chunks to load
#         chunk_dict: dictionary of index keys and filename values
#     """
#     data = {}
#     for k in chunk_indices:
#         data[k] = np.concatenate([feature.input_features(f) 
#                                   for f in chunk_dict[k]], axis=1)
#     return data

# def merge_clusters(data_dict, chunk_indices):
#     """
#     simplifies a lot of processes by concatenating the multiple chunks
#     into a single vector per-node
#     """
#     x = np.concatenate([data_dict[i] for i in chunk_indices],axis=0)
#     return x

def node_load_geotiff(filename, chunk_indices):
    pass

def node_load_hdf5(filenames, chunk_indices):
    pass

def all_image_data(image_dict, chunk_indices):
    """
    loads and concatenates a masked array containing all data
    on a particular node. For the purposes of computing statistics only.
    Not used in final output
    """
    xs = []
    ms = []
    for i in chunk_indices:
        img = image_dict[i].data()
        xv = img.data.reshape((-1,img.data.shape[2]))
        mv = img.mask.reshape((-1,img.mask.shape[2]))
        xs.append(xv)
        ms.append(mv)
    x = np.concatenate(xs,axis=0)
    m = np.concatenate(ms, axis=0)
    xm = np.ma.masked_array(x, mask=m)
    return xm


def write_data(chunk_dict, chunk_indices, transform, feature_name, output_dir):
    filenames = []
    for i in chunk_indices:
        data = chunk_dict[i].data()
        feature_vector = transform(data)
        filename = feature_name + "_{}.hdf5".format(i)
        full_path = os.path.join(output_dir, filename)
        feature.output_features(feature_vector.data, feature_vector.mask,
                                full_path)
        filenames.append(full_path)
    return filenames

def node_count(x):
    """
    note that this is a vector per dimension because x is masked
    """
    return x.count(axis=0)

def node_sum(x):
    result = np.ma.sum(x,axis=0)
    if np.ma.count_masked(result) != 0:
        raise ValueError("Too many missing values to compute sum")
    return result

def node_var(x):
    delta = x - np.ma.mean(x,axis=0)
    result = np.ma.sum(delta * delta,axis=0)
    if np.ma.count_masked(result) != 0:
        raise ValueError("Too many missing values to compute variance")
    return result.data

def node_outer(x):
    delta = x - np.ma.mean(x,axis=0)
    result = np.ma.dot(delta.T,delta)
    if np.ma.count_masked(result) != 0:
        raise ValueError("Too many missing values to compute outer product")
    return result.data

def node_sets(x):
    """
    works on a masked x
    """
    sets = [np.unique(np.ma.compressed(x[:,i])) for i in range(x.shape[1])]
    return sets

def centre(x, x_mean):
    return x - x_mean

def standardise(x, x_sd):
    return x / x_sd[np.newaxis, :]

def one_hot(x, x_set):
    out_dim_sizes = np.array([k.shape[0] for k in x_set])
    #The index points in the output array for each input dimension
    indices = np.hstack((np.array([0]), np.cumsum(out_dim_sizes)))
    total_dims = np.sum(out_dim_sizes)
    n = x.shape[0]
    out = np.empty((n,total_dims), dtype=float)
    out.fill(-0.5)
    out_mask = np.zeros((n,total_dims), dtype=bool)
    
    for dim_idx, dim_set in enumerate(x_set):
        dim_in = x[:, dim_idx]
        dim_mask = x.mask[:, dim_idx]
        dim_out = out[:,indices[dim_idx]:indices[dim_idx+1]]
        dim_out_mask = out_mask[:,indices[dim_idx]:indices[dim_idx+1]]
        dim_out_mask[:] = dim_mask[:,np.newaxis]
        for i, val in enumerate(dim_set):
            dim_out[:,i][dim_in==val] = 0.5
    result = np.ma.array(data=out, mask=out_mask)
    return result
        
