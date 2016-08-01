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

    print(mpiops.chunk_index, [v["x"].shape for v in extracted_chunks.values()])

    x = np.ma.concatenate([v["x"] for v in extracted_chunks.values()], axis=1)
    x_out, compose_settings = pipeline.compose_features(x, compose_settings)

    ################################################
    # TODO: Import all of the data once here
    # Use the function in mpiops.py
    ################################################

    X_list = mpiops.comm.allgather(x_out)
    X = np.ma.vstack(X_list)
    models = {}

    for algorithm, args in config.algdict.items():

        model, scores, Ys, EYs = pipeline.learn_model(X,
                                                      targets,
                                                      algorithm,
                                                      crossvalidate=True,
                                                      algorithm_params=args
                                                      )
        models[algorithm] = model

        # Outputs
        if mpiops.chunk_index == 0:

            outfile_scores = path.join(config.output_dir,
                                       config.name + "_" + algorithm +
                                       "_scores.json")

            geoio.export_scores(scores, Ys, EYs, outfile_scores)

    if mpiops.chunk_index == 0:

        outfile_state = path.join(config.output_dir, config.name + ".state")
        state_dict = {"models": models,
                      "image_settings": image_settings,
                      "compose_settings": compose_settings}

        with open(outfile_state, 'wb') as f:
            pickle.dump(state_dict, f)

    log.info("Finished!")


def main():
    if len(sys.argv) != 2:
        print("Usage: learningpipeline <configfile>")
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
