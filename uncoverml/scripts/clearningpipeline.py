"""
A pipeline for learning and validating models.
"""

import pickle
from collections import OrderedDict
import importlib.machinery
import logging
from os import path, mkdir, listdir
from glob import glob
import sys
import json
import copy
from itertools import product

import numpy as np

from uncoverml import geoio
from uncoverml import pipeline
from uncoverml import mpiops
from uncoverml import datatypes
from uncoverml import cluster

from uncoverml.scripts.learningpipeline import get_targets
from uncoverml.scripts.learningpipeline import gather_data
from uncoverml.scripts.learningpipeline import extract as ex_features
from uncoverml.scripts.predictionpipeline import extract as ex_all

# Logging
log = logging.getLogger(__name__)


def get_data(config):
    # Make the targets
    shapefile = path.join(config.data_dir, config.class_file)
    targets = mpiops.run_once(get_targets,
                              shapefile=shapefile,
                              fieldname='class',
                              folds=config.folds,
                              seed=config.crossval_seed)
    # should be ints
    targets.observations = targets.observations.astype(int)

    # training data is labelled
    extracted_chunks_t, image_settings = ex_features(targets, config)
    extracted_chunks_a = ex_all(0, 1, image_settings, config)

    compose_settings = datatypes.ComposeSettings(
        impute=config.impute,
        transform=config.transform,
        featurefraction=config.pca_frac,
        impute_mean=None,
        mean=None,
        sd=None,
        eigvals=None,
        eigvecs=None)

    # Grab the data
    X_t = gather_data(extracted_chunks_t, compose_settings)
    X_a = np.ma.concatenate([v["x"] for v in extracted_chunks_a.values()],
                            axis=1)
    X_a, compose_settings = pipeline.compose_features(X_a, compose_settings)

    no_missing = np.sum(X_t.mask, axis=1) == 0
    X_t = X_t[no_missing].data
    indices = np.arange(X_t.shape[0])[no_missing]
    classes = targets.observations[no_missing]

    no_missing = np.sum(X_a.mask, axis=1) == 0
    X_a = X_a[no_missing].data

    k = mpiops.comm.allreduce(np.amax(classes), op=mpiops.MPI.MAX)
    k = max(k, config.number_of_classes)

    X_a = np.concatenate([X_t, X_a], axis=0)
    return X_a, indices, classes, k, image_settings, compose_settings


def run_pipeline(config):
    X, indices, classes, k, image_settings, compose_settings = get_data(config)
    l = config.oversampling_factor
    training_data = cluster.TrainingData(indices, classes)
    C_init = cluster.initialise_centres(X, k, l, training_data)
    C_final, assignments = cluster.run_kmeans(X, C_init, k,
                                              training_data=training_data)
    export_model(C_final, image_settings, compose_settings, config)


def export_model(model, image_settings, compose_settings, config):
    outfile_state = path.join(config.output_dir,
                              config.name + ".cluster")
    state_dict = {"model": model,
                  "image_settings": image_settings,
                  "compose_settings": compose_settings}

    with open(outfile_state, 'wb') as f:
        pickle.dump(state_dict, f)


def main():
    if len(sys.argv) != 2:
        sys.exit(-1)
    logging.basicConfig(level=logging.INFO)
    config_filename = sys.argv[1]
    name = path.basename(config_filename).rstrip(".pipeline")
    config = importlib.machinery.SourceFileLoader(
        'config', config_filename).load_module()
    if not hasattr(config, 'name'):
        config.name = name
    config.output_dir = path.abspath(config.output_dir)
    run_pipeline(config)

if __name__ == "__main__":
    main()
