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

from uncoverml import image
from uncoverml import geoio
from uncoverml import pipeline

# Logging
log = logging.getLogger(__name__)


def image_subchunks(subchunk_index, n_subchunks, config):
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

        extracted_chunks = {}
        for tif in tifs:
            name = path.basename(tif)
            log.info("Processing {}.".format(name))
            image_source = geoio.RasterioImageSource(tif)
            x = pipeline.extract_subchunks(image_source, subchunk_index,
                                           n_subchunks, config.patchsize)
            extracted_chunks[name] = x
        extracted_chunks = OrderedDict(sorted(extracted_chunks.items(),
                                              key=lambda t: t[0]))
        results.append(extracted_chunks)
    return results


def get_image_spec(model, config):
    # temp workaround, we should have an image spec to check against
    nchannels = len(model.get_predict_tags())
    imagelike = glob(path.join(config.data_dir, "*.tif"))[0]
    template_image = image.Image(geoio.RasterioImageSource(imagelike))
    eff_shape = template_image.patched_shape(config.patchsize) + (nchannels,)
    eff_bbox = template_image.patched_bbox(config.patchsize)
    return eff_shape, eff_bbox


def render_partition(model, subchunk, n_subchunks, image_out,
                     transform_sets, config):

        extracted_chunk_sets = image_subchunks(subchunk, n_subchunks, config)
        x = pipeline.transform_features(extracted_chunk_sets, transform_sets)

        alg = config.algorithm
        log.info("Predicting targets for {}.".format(alg))

        y_star = pipeline.predict(x, model, interval=config.quantiles)
        image_out.write(y_star, subchunk)


def run_pipeline(config):

    outfile_state = path.join(config.output_dir,
                              config.name + "_" + config.algorithm + ".state")
    with open(outfile_state, 'rb') as f:
        state_dict = pickle.load(f)

    model = state_dict["model"]
    transform_sets = state_dict["transform_sets"]
    image_shape, image_bbox = get_image_spec(model, config)

    n_subchunks = max(1, round(1.0 / config.memory_fraction))
    log.info("Dividing node data into {} partitions".format(n_subchunks))

    outfile_tif = config.name + "_output_" + config.algorithm
    image_out = geoio.ImageWriter(image_shape, image_bbox, outfile_tif,
                                  n_subchunks, config.output_dir,
                                  band_tags=model.get_predict_tags())

    for i in range(n_subchunks):
        log.info("starting to render partition {}".format(i+1))
        render_partition(model, i, n_subchunks, image_out, transform_sets,
                         config)
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
