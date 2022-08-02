import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging

import uncoverml as ls

from uncoverml import predict

from uncoverml import features
from uncoverml import mpiops
from uncoverml import geoio
from uncoverml.models import apply_masked, modelmaps
from uncoverml.optimise.models import transformed_modelmaps
from uncoverml.krige import krig_dict
from uncoverml import transforms
from uncoverml.config import Config
from uncoverml import predict

from uncoverml.scripts import uncoverml as uncli


log = logging.getLogger(__name__)


def predict_save(model_or_cluster_file, partitions, mask, retain):
    with open(model_or_cluster_file, 'rb') as f:
        state_dict = joblib.load(f)

    model = state_dict["model"]
    config = state_dict["config"]
    config.cluster = True if os.path.splitext(model_or_cluster_file)[1] == '.cluster' \
        else False
    config.mask = mask if mask else config.mask
    if config.mask:
        config.retain = retain if retain else config.retain

        if not os.path.isfile(config.mask):
            config.mask = ''
            log.info('A mask was provided, but the file does not exist on '
                     'disc or is not a file.')

    config.n_subchunks = partitions
    if config.n_subchunks > 1:
        log.info("Memory contstraint forcing {} iterations "
                 "through data".format(config.n_subchunks))
    else:
        log.info("Using memory aggressively: dividing all data between nodes")

    image_shape, image_bbox, image_crs = ls.geoio.get_image_spec(model, config)

    outfile_tif = config.algorithm
    predict_tags = model.get_predict_tags()
    if not config.outbands:
        config.outbands = len(predict_tags)

    image_out = ls.geoio.ImageWriter(image_shape, image_bbox, image_crs,
                                     outfile_tif,
                                     config.n_subchunks, 'test_plots/',
                                     band_tags=predict_tags[
                                               0: min(len(predict_tags),
                                                      config.outbands)],
                                     **config.geotif_options)

    for i in range(config.n_subchunks):
        log.info("starting to render partition {}".format(i+1))
        ls.predict.render_partition(model, i, image_out, config)

    # explicitly close output rasters
    image_out.close()

    if config.cluster and config.cluster_analysis:
        if ls.mpiops.chunk_index == 0:
            ls.predict.final_cluster_analysis(config.n_classes,
                                              config.n_subchunks)

    # ls.predict.final_cluster_analysis(config.n_classes,
    #                                   config.n_subchunks)

    if config.thumbnails:
        image_out.output_thumbnails(config.thumbnails)
    log.info("Finished!")


def look_at_data(model_file):
    with open(model_file, 'rb') as f:
        state_dict = joblib.load(f)

    model = state_dict["model"]
    config = state_dict["config"]

    # noinspection PyProtectedMember
    targets, x = uncli._load_data(config, partitions=200)
    print(targets.shape)
    print(x.shape)


if __name__ == '__main__':
    model_file = 'gbquantile/gbquantiles.model'
    partitions = 200
    # predict_save(model_file, partitions, None, None)

    look_at_data(model_file)
