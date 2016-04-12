import numpy as np
import os.path
import time
import click
import ipyparallel as ipp
from uncoverml import feature
import logging

log = logging.getLogger(__name__)

def task_view(profile):
    c = ipp.Client(profile=profile) if profile is not None else ipp.Client()
    return c.load_balanced_view()

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
    for i, indices in enumerate(np.array_split(np.arange(nchunks),nworkers)):
        cmd = "chunk_indices = {}".format(indices.tolist())
        log.info("assigning engine {} chunks {}".format(i, indices))
        c.execute(cmd, targets=i)
    return c

def load_and_cat(chunk_indices, chunk_dict):
    """
    loads the data and concatenates it by dimension

    Parameters 
    ==========
        chunk_indices: the chunks to load
        chunk_dict: dictionary of index keys and filename values
    """
    data = {}
    for k in chunk_indices:
        feats = tuple(zip(*[feature.input_features(f) for f in chunk_dict[k]]))
        data[k] = (np.concatenate(feats[0], axis=1),
                   np.concatenate(feats[1], axis=1)
                   )
    return data

def merge_clusters(data_dict, chunk_indices):
    """
    simplifies a lot of processes by concatenating the multiple chunks
    into a single vector per-node
    """
    x = np.concatenate([data_dict[i] for i in chunk_indices],axis=0)
    return x


def print_async_progress(async_result, title):
    total_jobs = len(async_result)
    with click.progressbar(length=total_jobs, label=title) as bar:
        last_jobs_done = 0
        jobs_done = 0
        while jobs_done < total_jobs:
            jobs_done = async_result.progress
            if jobs_done > last_jobs_done:
                bar.update(jobs_done - last_jobs_done)
                last_jobs_done = jobs_done
            time.sleep(0.1)


def map(f, iterable, cluster_view=None):
    # Send off the jobs
    progress_title = "Processing Image Chunks"
    if cluster_view is not None:
        async_result = cluster_view.map(f, iterable, block=False)
        print_async_progress(async_result, progress_title)
        results = async_result.get()
    else:
        results = []
        with click.progressbar(length=len(iterable),
                               label=progress_title) as bar:
            for i in iterable:
                r = f(i)
                results.append(r)
                bar.update(1)
    return results


def write_data(data, transform, feature_name, output_dir):
    filenames = []
    for i, d in data.items():
        feature_vector = transform(d)
        filename = feature_name + "_{}.hdf5".format(i)
        full_path = os.path.join(output_dir, filename)
        feature.output_features(feature_vector, full_path)
        filenames.append(full_path)
    return filenames


# this is primarily because we cant pickle closures to use with above func.
def write_predict(data, model, target_name, output_dir):
    filenames = []

    for i, d in data.items():
        # FIXME deal with missing data in d[1]
        target_vector = model.predict(d[0])
        filename = target_name + "_{}.hdf5".format(i)
        full_path = os.path.join(output_dir, filename)
        feature.output_features(target_vector, full_path,
                                featname="predictions")
        filenames.append(full_path)
    return filenames


def node_count(x):
    return x.shape[0]

def node_sum(x):
    return np.sum(x,axis=0)

def node_var(x):
    return np.var(x,axis=0) * float(x.shape[0])

def node_outer(x):
    return np.cov(x,rowvar=0,bias=0) * float(x.shape[0])

def centre(x, x_mean):
    return x - x_mean

def standardise(x, x_variance):
    return x / x_variance[np.newaxis, :]

def map_over_data(f, cluster_view):
    all_results = {}
    results = cluster_view.apply(_engine_map, f)
    for r in results:
        all_results.update(r)
    # build the list
    n_chunks = len(results) #LOL hopefully
    result_list = [all_results[k] for k in range(n_chunks)]
    return result_list

def get_data():
    return data
