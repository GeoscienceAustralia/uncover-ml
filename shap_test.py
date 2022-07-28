import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uncoverml import config, predict
from uncoverml.scripts import uncoverml as uncli

import pandas as pd

with open('gbquantile/gbquantiles.model', 'rb') as f:
    state_dict = joblib.load(f)

def predict_for_shap(x_vals):
    # predictions = predict.predict(x_vals, model, interval=config.quantiles, lon_lat=targets_all.positions)
    predictions = predict.predict(x_vals, model)
    return predictions


if __name__ == '__main__':
    model = state_dict['model']
    config = state_dict['config']

    # noinspection PyProtectedMember
    targets_all, x_all = uncli._load_data(config, partitions=200)

    print('creating explainer')
    # explainer = shap.Explainer(predict_for_shap, x_all)
    masker = shap.maskers.Independent(x_all)
    explainer = shap.Explainer(predict_for_shap, masker)
    print('calculating shap values')
    shap_vals = explainer(x_all[:100])

    print('plotting shap values')

    # PLOTS TO ADD

    # Aggregate:
    #   - Benchmark, used to benchmark different explainers
    #   - Embedding
    #   - Group Difference
    #   - Monitoring
    #   - Partial Dependence

    # Individual:
    #   - Scatter
    #   - Waterfall


# ----------------------------- Aggregate Plots -----------------------------------------------

    # BEESWARM - WORKS
    # for idx in range(shap_vals.shape[2]):
    #     shap.summary_plot(shap_vals[:, :, idx], show=False)
    #     filename = 'gbquantile/summary_test_' + str(idx) + '.png'
    #     plt.savefig(filename)
    #     plt.clf()
    #
    # print('beeswarm complete')

    # BAR - WORKS
    # for idx in range(shap_vals.shape[2]):
    #     shap.plots.bar(shap_vals[:, :, idx], show=False)
    #     filename = 'gbquantile/bar_test_' + str(idx) + '.png'
    #     plt.savefig(filename)
    #     plt.clf()

    # DECISION - WORKS
    # for idx in range(shap_vals.shape[2]):
    #     shap.decision_plot(shap_vals[:, :, idx].base_values[0], shap_vals[:, :, idx].values, show=False)
    #     filename = 'gbquantile/dec_test_' + str(idx) + '.png'
    #     plt.savefig(filename)
    #     plt.clf()
    #
    # print('decision plot complete')

    # EMBEDDING PLOT
    shap.embedding_plot(0, shap_vals[:, :, 1], show=False)
    plt.savefig('embedding_test.png')
    plt.clf()

    print('embedding plot complete')


# ----------------------------- Individual Plots -----------------------------------------------

    # shap.plots.force(shap_vals[0].base_values[0], shap_vals[0].values)
    # print('force plot complete')
