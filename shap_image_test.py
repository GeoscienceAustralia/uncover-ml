import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import uncoverml as ls

from uncoverml import predict


def shap_image(model_or_cluster_file, partitions, mask, retain):
    with open(model_or_cluster_file, 'rb') as f:
        state_dict = joblib.load(f)

    model = state_dict["model"]
    config = state_dict["config"]

    config.mask = mask if mask else config.mask
    if config.mask:
        config.retain = retain if retain else config.retain

        if not isfile(config.mask):
            config.mask = ''
            print('A mask was provided, but the file does not exist on '
                     'disc or is not a file.')

    config.n_subchunks = partitions
    if config.n_subchunks > 1:
        print("Memory contstraint forcing {} iterations "
                 "through data".format(config.n_subchunks))
    else:
        print("Using memory aggressively: dividing all data between nodes")

    outfile_tif = config.algorithm
    predict_tags = model.get_predict_tags()
    if not config.outbands:
        config.outbands = len(predict_tags)

    image_out = ls.geoio.ImageWriter(image_shape, image_bbox, image_crs,
                                     outfile_tif,
                                     config.n_subchunks, config.output_dir,
                                     band_tags=predict_tags[
                                               0: min(len(predict_tags),
                                                      config.outbands)],
                                     **config.geotif_options)

    ###### SHAP CALCULATION GOES HERE #######

    image_out.close()
    if config.thumbnails:
        image_out.output_thumbnails(config.thumbnails)

    print("Finished! Total mem = {:.1f} GB".format(_total_gb()))