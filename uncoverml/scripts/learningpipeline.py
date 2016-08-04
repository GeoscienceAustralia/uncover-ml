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

# Logging
log = logging.getLogger(__name__)


def make_proc_dir(dirname):
    if not path.exists(dirname):
        mkdir(dirname)
        log.info("Made processed dir")


def get_targets(shapefile, fieldname, folds, seed):
    shape_infile = path.abspath(shapefile)
    lonlat, vals = geoio.load_shapefile(shape_infile, fieldname)
    targets = datatypes.CrossValTargets(lonlat, vals, folds, seed, sort=True)
    return targets


def extract(targets, feature_set):
    # Extract feats for training
    tifs = glob(path.join(feature_set['path'], "*.tif"))
    if len(tifs) == 0:
        log.fatal("No geotiffs found in {}!".format(feature_set['path']))
        sys.exit(-1)

    extracted_chunks = {}
    settings = {}
    for tif in tifs:
        name = path.basename(tif)
        log.info("Processing {}.".format(name))
        s = datatypes.ExtractSettings(onehot=feature_set['onehot'],
                                      x_sets=None,
                                      patchsize=feature_set['patchsize'])
        image_source = geoio.RasterioImageSource(tif)
        # x may be none, but everyone gets the same settings object
        x, s = pipeline.extract_features(image_source, targets, s)
        extracted_chunks[name] = x
        settings[name] = s
    result = OrderedDict(sorted(extracted_chunks.items(), key=lambda t: t[0]))
    return result, settings


class FeatureSet():
    def __init__(self, image_chunks, image_settings, compose_settings):
        self.image_chunks = image_chunks
        self.image_settings = image_settings
        self.compose_settings = compose_settings


def load_all_data(targets, config):
    features = []
    for f in config.feature_sets:
        # keys for these two are the filenames
        extracted_chunks_f, settings_f = extract(targets, f)
        compose_settings_f = datatypes.ComposeSettings(
            impute=f['impute'],
            transform=f['transform'],
            featurefraction=f['pca_frac'],
            impute_mean=None,
            mean=None,
            sd=None,
            eigvals=None,
            eigvecs=None)
        features.append(FeatureSet(extracted_chunks_f, settings_f,
                                   compose_settings_f))
    return features


def all_nodes_x(X):
    X_list = mpiops.comm.allgather(X)
    X = np.ma.vstack([k for k in X_list if k is not None])
    return X


def run_pipeline(config):

    # Make the targets
    targets = mpiops.run_once(get_targets,
                              shapefile=config.target_file,
                              fieldname=config.target_var,
                              folds=config.folds,
                              seed=config.crossval_seed)

    if config.export_targets:
        outfile_targets = path.join(config.output_dir,
                                    config.name + "_targets.hdf5")
        mpiops.run_once(geoio.write_targets, targets, outfile_targets)

    features = load_all_data(targets, config)

    if config.rank_features:
        for algorithm in sorted(config.algdict.keys()):
            measures, feature_list, scores = rank_features(features, targets,
                                                           algorithm, config)
            mpiops.run_once(export_feature_ranks, measures,
                            feature_list, scores, algorithm, config)

    print("extracting and composing per node:")
    full_node_x = extract_and_compose(features)
    print("gathering all data to all nodes:")
    X = all_nodes_x(full_node_x)
    print("now all data is at all nodes")
    for algorithm in sorted(config.algdict.keys()):
        args = config.algdict[algorithm]

        if config.cross_validate:
            crossval_results = pipeline.cross_validate(X, targets, algorithm,
                                                       args)
            mpiops.run_once(export_scores, crossval_results, algorithm, config)

        model = pipeline.learn_model(X, targets, algorithm, args)
        # mpiops.run_once(export_model, model, image_settings,
        #                 compose_settings, algorithm, config)


def rank_features(feature_sets, targets, algorithm, config):

    # Determine the importance of each feature
    all_feature_names = [k.rstrip(".tif") for f in feature_sets
                         for k in f.image_chunks]
    feature_scores = {}
    for name in all_feature_names:
        log.info("Computing {} feature importance of {}".format(algorithm,
                                                                name))
        full_node_x = extract_and_compose(feature_sets, leave_out=name)
        X = all_nodes_x(full_node_x)
        results = pipeline.cross_validate(X, targets, algorithm,
                                          config.algdict[algorithm])
        feature_scores[name] = results

    # Get the different types of score from one of the outputs
    # TODO make this not suck
    measures = list(next(feature_scores.values().__iter__()).scores.keys())
    features = sorted(feature_scores.keys())
    scores = np.empty((len(measures), len(features)))
    for m, measure in enumerate(measures):
        for f, feature in enumerate(features):
            scores[m, f] = feature_scores[feature].scores[measure]
    return measures, features, scores


def extract_and_compose(features, leave_out=None):
    composed_vectors = []
    # just test the first feature set because they all have same geometry
    no_data = (True in [k is None for k in features[0].image_chunks.values()])
    if no_data:
        return

    composed_vectors = []
    for f in features:
        chunks = [v for k, v in f.image_chunks.items()
                  if k != leave_out]
        x_f = np.ma.concatenate(chunks, axis=1)
        # prevent the leave-out extract from overwriting compose settings
        comp_settings = (copy.deepcopy(f.compose_settings)
                         if leave_out else f.compose_settings)
        x_out, comp_settings = pipeline.compose_features(x_f,
                                                         comp_settings)
        if not leave_out:
            f.compose_settings = comp_settings
        composed_vectors.append(x_out)

    # stack composed vectors
    full_node_x = np.ma.concatenate(composed_vectors, axis=1)
    return full_node_x
    # X_list = mpiops.comm.allgather(full_node_x)
    # X = np.ma.vstack([k for k in X_list if k is not None])
    # return X


def export_feature_ranks(measures, features, scores, algorithm, config):
    outfile_ranks = path.join(config.output_dir,
                              config.name + "_" + algorithm +
                              "_featureranks.json")

    score_listing = dict(scores={}, ranks={})
    for measure, measure_scores in zip(measures, scores):

        # Sort the scores
        scores = sorted(zip(features, measure_scores),
                        key=lambda s: s[1])
        sorted_features, sorted_scores = zip(*scores)

        # Store the results
        score_listing['scores'][measure] = sorted_scores
        score_listing['ranks'][measure] = sorted_features

    # Write the results out to a file
    with open(outfile_ranks, 'w') as output_file:
        json.dump(score_listing, output_file, sort_keys=True, indent=4)


def export_model(model, image_settings, compose_settings, algorithm, config):
    outfile_state = path.join(config.output_dir,
                              config.name + "_" + algorithm + ".state")
    state_dict = {"model": model,
                  "image_settings": image_settings,
                  "compose_settings": compose_settings}

    with open(outfile_state, 'wb') as f:
        pickle.dump(state_dict, f)


def export_scores(crossval_output, algorithm, config):

    outfile_scores = path.join(config.output_dir,
                               config.name + "_" + algorithm +
                               "_scores.json")
    geoio.export_scores(crossval_output.scores,
                        crossval_output.y_true,
                        crossval_output.y_pred,
                        outfile_scores)


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
