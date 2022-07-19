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

with open('gbquantile/gbquantiles.model', 'rb') as f:
    state_dict = joblib.load(f)

model = state_dict['model']
config = state_dict['config']

# noinspection PyProtectedMember
targets_all, x_all = uncli._load_data(config, partitions=200)
feature_list = list(config.intersected_features.values())


def predict_for_shap(x_vals):
    # predictions = predict.predict(x_vals, model, interval=config.quantiles, lon_lat=targets_all.positions)
    predictions = predict.predict(x_vals, model)
    return predictions


def plot_shap(shap_values, lon_lat):
    shap_vals_plot = shap_values[:, 0, 0].values
    cm = plt.cm.get_cmap('cool')

    plt.clf()
    sc = plt.scatter(lon_lat[:, 0], lon_lat[:, 1], s=10, c=shap_vals_plot, cmap=cm)
    plt.colorbar(sc)
    plt.savefig('test_plots/spatial_test.png')


if __name__ == '__main__':
    print('creating explainer')
    # explainer = shap.Explainer(predict_for_shap, x_all)
    masker = shap.maskers.Independent(x_all)
    explainer = shap.Explainer(predict_for_shap, masker)
    print('calculating shap values')
    shap_vals = explainer(x_all[:2000])

    plot_shap(shap_vals, targets_all.positions[:2000])

