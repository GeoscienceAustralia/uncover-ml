import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging

import rasterio

import uncoverml as ls

from uncoverml import features
from uncoverml import mpiops
from uncoverml import geoio
from uncoverml.models import apply_masked, modelmaps
from uncoverml.optimise.models import transformed_modelmaps
from uncoverml.krige import krig_dict
from uncoverml import transforms
from uncoverml.config import Config
from uncoverml import predict


log = logging.getLogger(__name__)


def single_predict(model_or_cluster_file, partitions, mask=None, retain=None):
    with open(model_or_cluster_file, 'rb') as f:
        state_dict = joblib.load(f)

    model = state_dict["model"]
    config = state_dict["config"]
    config.cluster = True if splitext(model_or_cluster_file)[1] == '.cluster' \
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

    log.info("starting to render partition {}".format(0))
    # noinspection PyProtectedMember
    x, feature_names = ls.predict._get_data(0, config)
    total_gb = mpiops.comm.allreduce(x.nbytes / 1e9)
    log.info("Loaded {:2.4f}GB of image data".format(total_gb))
    alg = config.algorithm
    log.info("Predicting targets for {}.".format(alg))
    y_star = ls.predict.predict(x, model, interval=config.quantiles,
                                lon_lat=_get_lon_lat(0, config))
    # cluster_analysis(x, y_star, subchunk, config, feature_names)
    image_out.write(y_star, subchunk)

    # explicitly close output rasters
    image_out.close()

    shap_calc(model, 'test_plots/gbquantile_prediction.tif', 0, config)

    log.info("Finished! Total mem = {:.1f} GB".format(_total_gb()))


def shap_calc(model, prediction_file, subchunk, config):

    def predict_for_shap(x_vals):
        # noinspection PyProtectedMember
        predictions = ls.predict.predict(x_vals, model, interval=config.quantiles,
                                      lon_lat=predict._get_lon_lat(subchunk, config))
        return predictions

    # noinspection PyProtectedMember
    x, feature_names = ls.predict._get_data(subchunk, config)
    masker = shap.maskers.Independent(x)
    explainer = shap.Explainer(predict_for_shap, masker)
    shap_vals = explainer(x)
    shap_vals_to_write = shap_vals[:, 0, 0]

    with rasterio.open(prediction_file, 'r') as img:
        im_arr = img.read()

    shap.image_plot(shap_vals_to_write, im_arr, show=False)
    plt.savefig('test_plots/image_test.png')
    plt.clf()


if __name__ == '__main__':
    model_file = 'gbquantile/gbquantiles.model'
    partitions = 200
    single_predict(model_file, partitions)
    log.info('Shap calculation complete')
