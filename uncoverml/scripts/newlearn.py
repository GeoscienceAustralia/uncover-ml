"""
A pipeline for learning and validating models.
"""

from collections import OrderedDict
import importlib.machinery
import logging
from os import path, mkdir, listdir
from glob import glob
import sys

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

    mpiops.run_once(make_proc_dir, config.proc_dir)

    if listdir(config.proc_dir):
        log.fatal("Output directory must be empty!")
        sys.exit(-1)

    # Make the targets
    shapefile = path.join(config.data_dir, config.target_file)
    targets = mpiops.run_once(get_targets,
                              shapefile=shapefile,
                              fieldname=config.target_var,
                              folds=config.folds,
                              seed=config.crossval_seed)

    extracted_chunks = extract(targets, config)

    compose_settings = datatypes.ComposeSettings(
        impute=config.impute,
        transform=config.transform,
        featurefraction=config.pca_frac,
        impute_mean=None,
        mean=None,
        sd=None,
        eigvals=None,
        eigvecs=None
        )

    x = np.ma.concatenate([v["x"] for v in extracted_chunks.values()], axis=1)
    x_out, compose_settings = pipeline.compose_features(x, compose_settings)

    for alg, args in config.algdict.items():

        X_list = mpiops.comm.gather(x_out, root=0)
        if mpiops.chunk_index == 0:
            model = pipeline.learn_model(X_list, targets, alg, cvindex=0,
                                         algorithm_params=args)
        model = mpiops.comm.bcast(model, root=0)

        # Test the model
        log.info("Predicting targets for {}.".format(alg))
        y_star = pipeline.predict(x_out, model, interval=None)

        Y_list = mpiops.comm.gather(y_star, root=0)
        if mpiops.chunk_index == 0:
            scores, Ys, EYs = pipeline.validate(targets, Y_list, cvindex=0)
            print(scores)

        log.info("Finished!")


def main():
    if len(sys.argv) != 2:
        print("Usage: learningpipeline <configfile>")
        sys.exit(-1)
    logging.basicConfig(level=logging.INFO)
    config_filename = sys.argv[1]
    config = importlib.machinery.SourceFileLoader(
        'config', config_filename).load_module()
    run_pipeline(config)

if __name__ == "__main__":
    main()
