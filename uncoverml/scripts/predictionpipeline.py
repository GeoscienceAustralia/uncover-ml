"""
A pipeline for learning and validating models.
"""

import pickle
from collections import OrderedDict
import importlib.machinery
import logging
from os import path, mkdir, listdir, getcwd
from glob import glob
import sys

import numpy as np

from uncoverml import image
from uncoverml import geoio
from uncoverml import pipeline

# Logging
log = logging.getLogger(__name__)


def extract(image_settings, config):
    # Extract feats for training
    tifs = glob(path.join(config.data_dir, "*.tif"))
    if len(tifs) == 0:
        log.fatal("No geotiffs found in {}!".format(config.data_dir))
        sys.exit(-1)

    extracted_chunks = {}
    for tif in tifs:
        name = path.basename(tif)
        settings = image_settings[name]
        log.info("Processing {}.".format(name))
        image_source = geoio.RasterioImageSource(tif)
        targets = None
        x, settings = pipeline.extract_features(image_source,
                                                targets, settings)
        d = {"x": x, "settings": settings}
        extracted_chunks[name] = d
    result = OrderedDict(sorted(extracted_chunks.items(), key=lambda t: t[0]))
    return result


def run_pipeline(config):

    outfile_state = path.join(config.output_dir, config.name
                              + "_" + config.algorithm + ".state")
    with open(outfile_state, 'rb') as f:
        state_dict = pickle.load(f)
    image_settings = state_dict["image_settings"]
    compose_settings = state_dict["compose_settings"]
    models = state_dict["model"]

    extracted_chunks = extract(image_settings, config)

    x = np.ma.concatenate([v["x"] for v in extracted_chunks.values()], axis=1)
    x_out, compose_settings = pipeline.compose_features(x, compose_settings)

    alg = config.algorithm
    model = models[alg]

    log.info("Predicting targets for {}.".format(alg))
    y_star = pipeline.predict(x_out, model, interval=None)

    # temp workaround
    imagelike = glob(path.join(config.data_dir, "*.tif"))[0]
    template_image = image.Image(geoio.RasterioImageSource(imagelike))
    eff_shape = template_image.patched_shape(config.patchsize)
    eff_bbox = template_image.patched_bbox(config.patchsize)

    outfile_tif = config.name + "_output_" + config.algorithm
    geoio.create_image(y_star,
                       shape=eff_shape,
                       bbox=eff_bbox,
                       name=outfile_tif,
                       outputdir=config.output_dir,
                       rgb=config.makergbtif)

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
