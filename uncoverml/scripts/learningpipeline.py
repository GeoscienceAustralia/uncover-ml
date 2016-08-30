"""
A pipeline for learning and validating models.
"""

import copy
import json
import logging
import sys
import pickle
from os import path

import numpy as np

from uncoverml import datatypes
from uncoverml import geoio
from uncoverml import mpiops
from uncoverml import pipeline
from uncoverml.validation import lower_is_better
from uncoverml.config import Config

# Logging
log = logging.getLogger(__name__)


def image_feature_sets(targets, config):

    def f(image_source):
        r = pipeline.extract_features(image_source, targets,
                                      config.n_subchunks, config.patchsize)
        return r
    result = geoio._iterate_sources(f, config)
    return result


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
    targets = geoio.load_targets(shapefile=config.target_file,
                                 targetfield=config.target_property)
    # We're doing local models at the moment
    targets_all = gather_targets(targets)

    # Get the image chunks and their associated transforms
    image_chunk_sets = image_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    if config.rank_features:
        measures, features, scores = local_rank_features(image_chunk_sets,
                                                         transform_sets,
                                                         targets_all,
                                                         config)
        mpiops.run_once(export_feature_ranks, measures, features,
                        scores, config)

    x = pipeline.transform_features(image_chunk_sets, transform_sets,
                                    config.final_transform)
    # learn the model
    # local models need all data
    x_all = gather_features(x)

    if config.cross_validate:
        crossval_results = pipeline.local_crossval(x_all, targets_all, config)
        mpiops.run_once(export_crossval, crossval_results, config)

    model = pipeline.local_learn_model(x, targets, config)
    mpiops.run_once(export_model, model, config)


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
        final_transform_leaveout = copy.deepcopy(config.final_transform)
        image_chunks_leaveout = [copy.copy(k) for k in image_chunk_sets]
        for c in image_chunks_leaveout:
            if name in c:
                c.pop(name)

        fname = name.rstrip(".tif")
        log.info("Computing {} feature importance of {}"
                 .format(config.algorithm, fname))

        x = pipeline.transform_features(image_chunks_leaveout,
                                        transform_sets_leaveout,
                                        final_transform_leaveout)
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


def export_model(model, config):
    outfile_state = path.join(config.output_dir,
                              config.name + ".model")
    state_dict = {"model": model,
                  "config": config}
    with open(outfile_state, 'wb') as f:
        pickle.dump(state_dict, f)


def export_crossval(crossval_output, config):
    import tables as hdf
    outfile_scores = path.join(config.output_dir,
                               config.name + "_scores.json")
    with open(outfile_scores, 'w') as f:
        json.dump(crossval_output.scores, f, sort_keys=True, indent=4)

    outfile_results = path.join(config.output_dir,
                                config.name + "_results.hdf5")
    with hdf.open_file(outfile_results, 'w') as f:
        for fld, v in crossval_output.y_pred.items():
            label = "_".join(fld.split())
            f.create_array("/", label, obj=v.data)
            f.create_array("/", label + "_mask", obj=v.mask)
        f.create_array("/", "y_true", obj=crossval_output.y_true)


def main():
    if len(sys.argv) != 2:
        sys.exit(-1)
    logging.basicConfig(level=logging.INFO)
    config_filename = sys.argv[1]
    config = Config(config_filename)
    run_pipeline(config)

if __name__ == "__main__":
    main()
