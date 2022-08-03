import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    feature_list = list(config.intersected_features.values())

    print('creating explainer')
    # explainer = shap.Explainer(predict_for_shap, x_all)
    masker = shap.maskers.Independent(x_all)
    explainer = shap.Explainer(predict_for_shap, masker)
    print('calculating shap values')
    shap_vals = explainer(x_all[:1000])
    # shap_interaction_vals = explainer.shap_interaction_values(x_all[:1000])

    print('plotting shap values')

    # PLOTS TO ADD

    # Aggregate:
    #   - Partial Dependence

    # Individual:
    #   - Waterfall


# ----------------------------- Aggregate Plots -----------------------------------------------

    # BEESWARM - WORKS
    # for idx in range(shap_vals.shape[2]):
    #     shap.summary_plot(shap_vals[:, :, idx].values, features=shap_vals[:, :, idx].data, feature_names = feature_list, show=False)
    #     if idx == 0:
    #         ax = plt.gca()
    #         ax.set_xlim(-0.5, 0.5)
    #
    #     filename = 'test_plots/summary_test_' + str(idx) + '.png'
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

    # GROUP FORCE - WORKS
    # for idx in range(shap_vals.shape[2]):
    #     shap.force_plot(shap_vals[:, :, idx].base_values[0], shap_vals[:, :, idx].values,
    #                     features=shap_vals[:, :, idx].data, feature_names = feature_list, show=False)
    #     filename = 'test_plots/group_force_test_' + str(idx) + '.png'
    #     plt.savefig(filename)
    #     plt.clf()
    #
    # print('group force plot complete')

# ----------------------------- Feature-based Plots -----------------------------------------------


    # EMBEDDING PLOT - WORKS - MIGHT NOT BE USEFUL
    # for cov_idx in range(shap_vals.shape[1]):
    #     shap.embedding_plot(cov_idx, shap_vals[:, :, 0].values, show=False)
    #     filename = 'gbquantile/embedding_test_' + str(cov_idx) + '.png'
    #     plt.savefig(filename)
    #     plt.clf()
    #
    # print('embedding plot complete')

    # DEPENDENCE PLOT - WORKS
    # shap.dependence_plot(0, shap_vals[:, :, 1].values, shap_vals[:, :, 1].data, show=False)
    # plt.savefig('dependence_test.png')
    # print('dependence plot complete')

# ------------------------------- Main Plots ---------------------------------------------------

    # BEESWARM - SUBPLOTS
    # fig, axes = plt.subplots(nrows=1, ncols=shap_vals.shape[2])
    # for idx in range(shap_vals.shape[2]):
    #     plt.subplot(1, shap_vals.shape[2], idx+1)
    #     current_frame = plt.gca()
    #     current_frame.axes.get_yaxis().set_visible(False)
    #     shap.summary_plot(shap_vals[:, :, idx].values, features=shap_vals[:, :, idx].data, feature_names=feature_list,
    #                      show=False, plot_size=None)
    #     if idx == 0:
    #         ax = plt.gca()
    #         ax.set_xlim(-0.5, 0.5)
    #
    # plt.tight_layout()
    # plt.savefig('test_plots/beeswarm_test.png')
    # plt.clf()
    # print('Summary plot complete')

    # BAR - SUBPLOTS
    # fig, axes = plt.subplots(nrows=1, ncols=shap_vals.shape[2])
    # for idx in range(shap_vals.shape[2]):
    #     plt.subplot(1, shap_vals.shape[2], idx + 1)
    #     current_frame = plt.gca()
    #     current_frame.axes.get_yaxis().set_visible(False)
    #     shap.plots.bar(shap_vals[:, :, idx], show=False)
    #
    # plt.tight_layout()
    # plt.savefig('test_plots/bar_test.png')
    # plt.clf()
    # print('Bar plot complete')

    # DECISION
    # shap.decision_plot(shap_vals[:, :, 0].base_values[0], shap_vals[:, :, 0].values, feature_names=feature_list,
    #                    show=False)
    # plt.tight_layout()
    # plt.savefig('test_plots/decision_pred_test.png')
    # plt.clf()
    # print('Decision plot complete')

    # GROUP FORCE - SUBPLOTS
    # fig, axes = plt.subplots(nrows=1, ncols=shap_vals.shape[2])
    # for idx in range(shap_vals.shape[2]):
    #     plt.subplot(1, shap_vals.shape[2], idx + 1)
    #     current_frame = plt.gca()
    #     current_frame.axes.get_yaxis().set_visible(False)
    #     shap.force_plot(shap_vals[:, :, idx].base_values[0], shap_vals[:, :, idx].values, shap_vals[:, :, idx].data,
    #                     feature_names=feature_list, show=False)
    #
    # plt.tight_layout()
    # plt.savefig('test_plots/group_force_test.png')
    # plt.clf()
    # print('Group force plot complete')

    # A few scatter plots
    # for feature_idx in range(9):
    #     shap.dependence_plot(feature_idx, shap_vals[:, :, 1].values, shap_vals[:, :, 1].data,
    #                          feature_names=feature_list, show=False)
    #     plt.tight_layout()
    #     filename = 'test_plots/dependence_plot_' + str(feature_idx) + '.png'
    #     plt.savefig(filename)
    #     plt.clf()
    #
    # print('Scatters plots complete')

    # Correlation matrix plot
    corr_matrix = pd.DataFrame(shap_vals[:, :, 0].values, columns=feature_list).corr()

    plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
    sns.set(font_scale=1.2)
    sns.heatmap(corr_matrix, cmap='coolwarm', fmt='.1g', annot=False)

    plt.savefig('test_plots/shap_correlation.png', dpi=200, bbox_inches='tight')
    print('correlation plot complete')

