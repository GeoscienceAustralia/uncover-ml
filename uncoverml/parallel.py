import numpy as np
import os.path
import time
import click
import ipyparallel as ipp
from uncoverml import feature
import logging

log = logging.getLogger(__name__)

# Very basic memoisation of the data to prevent us reloading it every time
# we call a function on the workers
data = {}

# Module-level constant set at initialisation that
# assigns chunks per worker
__chunk_indices = []

def chunk_indices():
    return __chunk_indices


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
    
    # Assign the chunks to workers
    nworkers = len(c)
    for i, indices in enumerate(np.array_split(np.arange(nchunks),nworkers)):
        cmd = "parallel.__chunk_indices = {}".format(indices.tolist())
        log.info("assigning engine {} chunks {}".format(i, indices))
        c.execute(cmd, targets=i)
    return c

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

def load_data(chunk_dict):
    """
    Loads data into the module-level cache (respecting the index on the 
    current engine)

    Parameters 
    ==========
        chunk_dict: dictionary of index keys and filename values

    """
    for k in __chunk_indices:
        data[k] = np.concatenate([feature.input_features(f) for f in chunk_dict[k]],
                                 axis=1)


def write_data(transform, feature_name, output_dir):
    filenames = []
    for i,d in data.items():
        feature_vector = transform(d)
        filename = feature_name + "_{}.hdf5".format(i)
        full_path = os.path.join(output_dir, filename)
        feature.output_features(feature_vector, full_path)
        filenames.append(full_path)
    return filenames


def _engine_map(f):
    results = {k:f(d) for k,d in data.items()}
    return results


def mean(x):
    x_sum = np.sum(x, axis=0)
    x_n = x.shape[0]
    return x_sum, x_n

def cov(x, mean):
    """
    outer product (unnormalized covariance) and number of data points
    in x used for computing a whitening transform
    """
    x_outer = np.cov(x - mean,rowvar=0,bias=0) * float(x.shape[0])
    return x_outer

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
