"""
A pipeline for learning and validating models.
"""

import copy
import importlib.machinery
import json
import logging
import sys
import pickle
from collections import OrderedDict
from os import path, mkdir
from glob import glob

import numpy as np

from uncoverml import datatypes
from uncoverml import geoio
from uncoverml import mpiops
from uncoverml import pipeline
from uncoverml import transforms
from uncoverml.validation import lower_is_better

# Logging
log = logging.getLogger(__name__)


def make_proc_dir(dirname):
    if not path.exists(dirname):
        mkdir(dirname)
        log.info("Made processed dir")


def image_transform_sets(config):
    transform_sets = []
    for i in range(2):
        # fake it for now
        # load the transforms
        transform_set = transforms.TransformSet()

        # transform_set.image_transforms.append(transforms.OneHotTransform())
        transform_set.imputer = transforms.MeanImputer()
        transform_set.global_transforms.append(transforms.CentreTransform())
        transform_set.global_transforms.append(
            transforms.StandardiseTransform())
        # transform_set.global_transforms.append(
        # transforms.WhitenTransform(keep_fraction=0.5))
        transform_sets.append(transform_set)
    return transform_sets


def image_feature_sets(targets, config):
    n_subchunks = max(1, round(1.0 / config.memory_fraction))

    # fake it for the moment
    results = []
    for i in range(2):
        # Extract feats for training
        tifs = glob(path.join(config.data_dir, "*.tif"))
        n_on_2 = int(round(len(tifs)/2))
        if i == 0:
            tifs = tifs[0:n_on_2]
        else:
            tifs = tifs[n_on_2:]

        if len(tifs) == 0:
            log.fatal("No geotiffs found in {}!".format(config.data_dir))
            sys.exit(-1)

        extracted_chunks = {}
        for tif in tifs:
            name = path.basename(tif)
            log.info("Processing {}.".format(name))
            image_source = geoio.RasterioImageSource(tif)
            x = pipeline.extract_features(image_source, targets, n_subchunks,
                                          config.patchsize)
            extracted_chunks[name] = x
        extracted_chunks = OrderedDict(sorted(
            extracted_chunks.items(), key=lambda t: t[0]))

        results.append(extracted_chunks)
    return results


def gather_targets(targets):
    y = np.ma.concatenate(mpiops.comm.allgather(targets.observations), axis=0)
    p = np.ma.concatenate(mpiops.comm.allgather(targets.positions), axis=0)
    d = {}
    keys = sorted(list(targets.fields.keys()))
    for k in keys:
        d[k] = np.ma.concatenate(mpiops.comm.allgather(targets.fields[k]),
                                 axis=0)
    result = datatypes.Targets(p, y, othervals=d)
    return result


def gather_features(x):
    x_all = np.ma.vstack(mpiops.comm.allgather(x))
    return x_all


def run_pipeline(config):

    # Make the targets
    shapefile = path.join(config.data_dir, config.target_file)
    targets = geoio.load_targets(shapefile=shapefile,
                                 targetfield=config.target_var)
    # We're doing local models at the moment
    targets_all = gather_targets(targets)

    # keys for these two are the filenames
    image_chunk_sets = image_feature_sets(targets, config)
    transform_sets = image_transform_sets(config)

    if config.rank_features:
        measures, features, scores = local_rank_features(image_chunk_sets,
                                                         transform_sets,
                                                         targets_all,
                                                         config)
        mpiops.run_once(export_feature_ranks, measures, features,
                        scores, config)

    x = pipeline.transform_features(image_chunk_sets, transform_sets)
    # learn the model
    # local models need all data
    x_all = gather_features(x)

    if config.cross_validate:
        crossval_results = pipeline.local_crossval(x_all, targets_all, config)
        mpiops.run_once(export_scores, crossval_results, config)

    model = pipeline.local_learn_model(x, targets, config)
    mpiops.run_once(export_model, model, transform_sets, config)


def local_rank_features(image_chunk_sets, transform_sets, targets_all, config):

    # Determine the importance of each feature
    feature_scores = {}
    # get all the images
    all_names = []
    for c in image_chunk_sets:
        all_names.extend(list(c.keys()))
    all_names = sorted(list(set(all_names)))  # make unique

    for name in all_names:
        transform_sets_leaveout = copy.deepcopy(transform_sets)
        image_chunks_leaveout = [copy.copy(k) for k in image_chunk_sets]
        for c in image_chunks_leaveout:
            if name in c:
                c.pop(name)

        fname = name.rstrip(".tif")
        log.info("Computing {} feature importance of {}"
                 .format(config.algorithm, fname))

        x = transform_features(image_chunks_leaveout,
                               transform_sets_leaveout)
        x_all = gather_features(x)

        results = pipeline.local_crossval(x_all, targets_all, config)
        feature_scores[fname] = results

    # Get the different types of score from one of the outputs
    # TODO make this not suck
    measures = list(next(feature_scores.values().__iter__()).scores.keys())
    features = sorted(feature_scores.keys())
    scores = np.empty((len(measures), len(features)))
    for m, measure in enumerate(measures):
        for f, feature in enumerate(features):
            scores[m, f] = feature_scores[feature].scores[measure]
    return measures, features, scores


def export_feature_ranks(measures, features, scores, config):
    outfile_ranks = path.join(config.output_dir,
                              config.name + "_" + config.algorithm +
                              "_featureranks.json")

    score_listing = dict(scores={}, ranks={})
    for measure, measure_scores in zip(measures, scores):

        # Sort the scores
        scores = sorted(zip(features, measure_scores),
                        key=lambda s: s[1])
        if measure in lower_is_better:
            scores.reverse()
        sorted_features, sorted_scores = zip(*scores)

        # Store the results
        score_listing['scores'][measure] = sorted_scores
        score_listing['ranks'][measure] = sorted_features

    # Write the results out to a file
    with open(outfile_ranks, 'w') as output_file:
        json.dump(score_listing, output_file, sort_keys=True, indent=4)


def export_model(model, transform_sets, config):
    outfile_state = path.join(config.output_dir,
                              config.name + "_" + config.algorithm + ".state")
    state_dict = {"model": model,
                  "transform_sets": transform_sets}

    with open(outfile_state, 'wb') as f:
        pickle.dump(state_dict, f)


def export_scores(crossval_output, config):

    outfile_scores = path.join(config.output_dir,
                               config.name + "_" + config.algorithm +
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
