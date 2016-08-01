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


def extract(targets, config):
    # Extract feats for training
    tifs = glob(path.join(config.data_dir, "*.tif"))
    if len(tifs) == 0:
        log.fatal("No geotiffs found in {}!".format(config.data_dir))
        sys.exit(-1)

    extracted_chunks = {}
    for tif in tifs:
        name = path.basename(tif)
        log.info("Processing {}.".format(name))
        settings = datatypes.ExtractSettings(onehot=config.onehot,
                                             x_sets=None,
                                             patchsize=config.patchsize)
        image_source = geoio.RasterioImageSource(tif)
        x, settings = pipeline.extract_features(image_source,
                                                targets, settings)
        d = {"x": x, "settings": settings}
        extracted_chunks[name] = d
    result = OrderedDict(sorted(extracted_chunks.items(), key=lambda t: t[0]))
    return result


def run_pipeline(config):

    # Make the targets
    shapefile = path.join(config.data_dir, config.target_file)
    targets = mpiops.run_once(get_targets,
                              shapefile=shapefile,
                              fieldname=config.target_var,
                              folds=config.folds,
                              seed=config.crossval_seed)

    if config.export_targets:
        outfile_targets = path.join(config.output_dir,
                                    config.name + "_targets.hdf5")
        mpiops.run_once(geoio.write_targets, targets, outfile_targets)

    # keys for these two are the filenames
    extracted_chunks = extract(targets, config)
    image_settings = {k: v["settings"] for k, v in extracted_chunks.items()}

    compose_settings = datatypes.ComposeSettings(
        impute=config.impute,
        transform=config.transform,
        featurefraction=config.pca_frac,
        impute_mean=None,
        mean=None,
        sd=None,
        eigvals=None,
        eigvecs=None)

    algorithm = 'svr'
    rank_features(extracted_chunks, targets, algorithm, compose_settings,
                  config)


def rank_features(extracted_chunks, targets, algorithm, compose_settings,
                  config):

    # Determine the importance of each feature
    feature_scores = {}
    for name in extracted_chunks:
        dict_missing = dict(extracted_chunks)
        del dict_missing[name]

        fname = name.rstrip(".tif")
        log.info("Computing {} feature importance of {}".format(algorithm,
                                                                fname))

        compose_missing = copy.deepcopy(compose_settings)
        out = predict_and_score(dict_missing, targets, algorithm,
                                compose_missing, config)
        feature_scores[fname] = out

    # Get the different types of score from one of the outputs
    # TODO make this not suck
    measures = list(next(feature_scores.values().__iter__()).scores.keys())
    features = sorted(feature_scores.keys())
    scores = np.empty((len(measures), len(features)))
    for m, measure in enumerate(measures):
        for f, feature in enumerate(features):
            scores[m, f] = feature_scores[feature].scores[measure]

    # Save the feature scores to a file
    dump_feature_ranks(measures, features, scores, "scores.json")


def dump_feature_ranks(measures, features, scores, filename):

    score_listing = dict(scores={}, ranks={})
    for measure, measure_scores in zip(measures, scores):

        # Sort the scores
        scores = sorted(list(zip(features, measure_scores)),
                        key=lambda s: s[1])
        sorted_features, sorted_scores = list(zip(*scores))

        # Store the results
        score_listing['scores'][measure] = sorted_scores
        score_listing['ranks'][measure] = sorted_features

    # Write the results out to a file
    with open(filename, 'w') as output_file:
        json.dump(score_listing, output_file)


def predict_and_score(extracted_chunks, targets, algorithm,
                      compose_settings, config):

    x = np.ma.concatenate([v["x"] for v in extracted_chunks.values()], axis=1)
    x_out, compose_settings = pipeline.compose_features(x, compose_settings)

    X_list = mpiops.comm.allgather(x_out)
    X = np.ma.vstack(X_list)
    args = config.algdict[algorithm]
    out = pipeline.learn_model(X, targets, algorithm, crossvalidate=True,
                               algorithm_params=args)
    return out


def dump_state(config, models, image_settings, compose_settings):
    if mpiops.chunk_index == 0:
        outfile_state = path.join(config.output_dir,
                                  config.name + ".state")
        state_dict = {"models": models,
                      "image_settings": image_settings,
                      "compose_settings": compose_settings}

        with open(outfile_state, 'wb') as f:
            pickle.dump(state_dict, f)


def dump_outputs(outputs, config):

    for algorithm, model_out in outputs.items():
        # Outputs
        if mpiops.chunk_index == 0:
            outfile_scores = path.join(config.output_dir,
                                       config.name + "_" + algorithm +
                                       "_scores.json")
            geoio.export_scores(model_out.scores,
                                model_out.y_true,
                                model_out.y_pred,
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
