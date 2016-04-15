import numpy as np
import os.path
import ipyparallel as ipp
from uncoverml import feature
import logging

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

def write_data(data_dict, transform, feature_name, output_dir):
    filenames = []
    for i in data_dict:
        data = data_dict[i]
        feature_vector = transform(data)
        filename = feature_name + "_{}.hdf5".format(i)
        full_path = os.path.join(output_dir, filename)
        feature.output_features(feature_vector, full_path)
        filenames.append(full_path)
    return filenames


def node_count(x):
    """
    note that this is a vector per dimension because x is masked
    """
    return x.count(axis=0)

def node_full_count(x):
    """
    total number of points including missing
    """
    return x.shape[0]

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

def impute_with_mean(x, mean):
    xi = np.ma.masked_array(data=np.copy(x.data), 
                            mask=np.zeros_like(x.data,dtype=bool))
    xi.data[x.mask] = np.broadcast_to(mean,x.shape)[x.mask]
    return xi

def one_hot(x, x_set):
    assert x.data.shape == x.mask.shape
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
        
