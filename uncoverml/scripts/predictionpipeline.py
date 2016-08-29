"""
A pipeline for learning and validating models.
"""

import pickle
import logging
from os import path
import sys

from uncoverml import image
from uncoverml import geoio
from uncoverml import pipeline
from uncoverml.config import Config

# Logging
log = logging.getLogger(__name__)


def image_subchunks(subchunk_index, config):

    def f(image_source):
        r = pipeline.extract_subchunks(image_source, subchunk_index,
                                       config.n_subchunks, config.patchsize)
        return r
    result = geoio._iterate_sources(f, config)
    return result


def get_image_spec(model, config):
    # temp workaround, we should have an image spec to check against
    nchannels = len(model.get_predict_tags())
    imagelike = config.feature_sets[0].files[0]
    template_image = image.Image(geoio.RasterioImageSource(imagelike))
    eff_shape = template_image.patched_shape(config.patchsize) + (nchannels,)
    eff_bbox = template_image.patched_bbox(config.patchsize)
    return eff_shape, eff_bbox


def render_partition(model, subchunk, image_out, config):

        extracted_chunk_sets = image_subchunks(subchunk, config)
        transform_sets = [k.transform_set for k in config.feature_sets]
        x = pipeline.transform_features(extracted_chunk_sets, transform_sets,
                                        config.final_transform)
        alg = config.algorithm
        log.info("Predicting targets for {}.".format(alg))

        y_star = pipeline.predict(x, model, interval=config.quantiles)
        image_out.write(y_star, subchunk)


def run_pipeline(model, config):

    image_shape, image_bbox = get_image_spec(model, config)

    outfile_tif = config.name + "_" + config.algorithm
    image_out = geoio.ImageWriter(image_shape, image_bbox, outfile_tif,
                                  config.n_subchunks, config.output_dir,
                                  band_tags=model.get_predict_tags())

    for i in range(config.n_subchunks):
        log.info("starting to render partition {}".format(i+1))
        render_partition(model, i, image_out, config)
    log.info("Finished!")


def main():
    if len(sys.argv) != 2:
        print("Usage: predictionpipeline <statefile>")
        sys.exit(-1)
    logging.basicConfig(level=logging.INFO)
    state_filename = sys.argv[1]
    with open(state_filename, 'rb') as f:
        state_dict = pickle.load(f)

    model = state_dict["model"]
    config = state_dict["config"]

    run_pipeline(model, config)

if __name__ == "__main__":
    main()
