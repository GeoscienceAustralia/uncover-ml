"""
A pipeline for learning clusters in input images
"""

import logging
import sys
import pickle
from os import path

import numpy as np

# from uncoverml import datatypes
from uncoverml import geoio
from uncoverml import mpiops
from uncoverml import pipeline
from uncoverml import cluster
from uncoverml.config import Config

# Logging
log = logging.getLogger(__name__)


def all_feature_sets(targets, config):

    def f(image_source):
        r_t = pipeline.extract_features(image_source, targets, n_subchunks=1,
                                        patchsize=config.patchsize)
        r_a = pipeline.extract_subchunks(image_source, subchunk_index=0,
                                         n_subchunks=1,
                                         patchsize=config.patchsize)
        r = np.ma.concatenate([r_t, r_a], axis=0)
        return r
    result = geoio._iterate_sources(f, config)
    return result


def remove_missing(targets, x):
    log.info("Stripping out missing data")
    no_missing_x = np.sum(x.mask, axis=1) == 0
    x = x.data[no_missing_x]

    # remove labels that correspond to data missing in x
    classes = targets.observations
    no_missing_y = no_missing_x[0:(classes.shape[0])]
    classes = classes[:, np.newaxis][no_missing_y]
    classes = classes.flatten()
    return x, classes


def compute_n_classes(classes, config):
    k = mpiops.comm.allreduce(np.amax(classes), op=mpiops.MPI.MAX)
    k = max(k, config.n_classes)
    return k


def run_pipeline(config):

    # make sure we're clear that we're clustering
    config.algorithm = config.clustering_algorithm
    # Get the taregts
    targets = geoio.load_targets(shapefile=config.class_file,
                                 targetfield=config.class_property)

    # Get the image chunks and their associated transforms
    image_chunk_sets = all_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    x = pipeline.transform_features(image_chunk_sets, transform_sets,
                                    config.final_transform)

    x, classes = remove_missing(targets, x)
    indices = np.arange(classes.shape[0], dtype=int)

    k = compute_n_classes(classes, config)
    model = cluster.KMeans(k, config.oversample_factor)
    log.info("Clustering image")
    model.learn(x, indices, classes)
    mpiops.run_once(export_model, model, config)


def export_model(model, config):
    outfile_state = path.join(config.output_dir,
                              config.name + ".cluster")
    state_dict = {"model": model,
                  "config": config}
    with open(outfile_state, 'wb') as f:
        pickle.dump(state_dict, f)


def main():
    if len(sys.argv) != 2:
        sys.exit(-1)
    logging.basicConfig(level=logging.INFO)
    config_filename = sys.argv[1]
    config = Config(config_filename)
    run_pipeline(config)

if __name__ == "__main__":
    main()
