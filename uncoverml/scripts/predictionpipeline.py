"""
A pipeline for learning and validating models.
"""

import pickle
from collections import OrderedDict
import importlib.machinery
import logging
from os import path
from glob import glob
import sys

import numpy as np

from uncoverml import image
from uncoverml import geoio
from uncoverml import pipeline

# Logging
log = logging.getLogger(__name__)


def extract(subchunk_index, n_subchunks, image_settings, config):
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
        x = pipeline.extract_subchunks(image_source, subchunk_index,
                                       n_subchunks, settings)
        d = {"x": x}
        extracted_chunks[name] = d
    result = OrderedDict(sorted(extracted_chunks.items(), key=lambda t: t[0]))
    return result


def render_partition(model, subchunk, n_subchunks, image_out,
                     image_settings, compose_settings, config):
        extracted_chunks = extract(subchunk, n_subchunks,
                                   image_settings, config)
        x = np.ma.concatenate([v["x"] for v in extracted_chunks.values()],
                              axis=1)
        x_out, compose_settings = pipeline.compose_features(x,
                                                            compose_settings)
        alg = config.algorithm
        log.info("Predicting targets for {}.".format(alg))
        y_star = pipeline.predict(x_out, model, interval=config.quantiles)
        image_out.write(y_star, subchunk)


def run_pipeline(config):

    outfile_state = path.join(config.output_dir,
                              config.name + "_" + config.algorithm + ".state")
    with open(outfile_state, 'rb') as f:
        state_dict = pickle.load(f)

    image_settings = state_dict["image_settings"]
    compose_settings = state_dict["compose_settings"]
    model = state_dict["model"]

    nchannels = pipeline.predict_channels(model, config.quantiles)

    # temp workaround
    imagelike = glob(path.join(config.data_dir, "*.tif"))[0]
    template_image = image.Image(geoio.RasterioImageSource(imagelike))
    eff_shape = template_image.patched_shape(config.patchsize) + (nchannels,)
    eff_bbox = template_image.patched_bbox(config.patchsize)

    n_subchunks = max(1, round(1.0 / config.memory_fraction))
    log.info("Dividing node data into {} partitions".format(n_subchunks))

    outfile_tif = config.name + "_output_" + config.algorithm
    image_out = geoio.ImageWriter(eff_shape, eff_bbox, outfile_tif,
                                  n_subchunks, config.output_dir)

    for i in range(n_subchunks):
        render_partition(model, i, n_subchunks, image_out, image_settings,
                         compose_settings, config)
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
