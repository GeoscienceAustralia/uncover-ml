import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uncoverml import config, predict
from uncoverml.scripts import uncoverml as uncli

import pandas as pd

with open('ref_xgb/reference_xgboost.model', 'rb') as f:
    state_dict = joblib.load(f)

model = state_dict['model']
config = config.Config('configs/reference_xgboost.yaml')

# noinspection PyProtectedMember
targets_all, x_all = uncli._load_data(config, partitions=4)


def predict_for_shap(x_vals):
    # predictions = predict.predict(x_vals, model, interval=config.quantiles, lon_lat=targets_all.positions)
    predictions = predict.predict(x_vals, model)
    return predictions


if __name__ == '__main__':
    print('creating explainer')
    # explainer = shap.Explainer(predict_for_shap, x_all)
    masker = shap.maskers.Independent(x_all)
    explainer = shap.Explainer(predict_for_shap, masker)
    print('calculating shap values')
    shap_vals = explainer(x_all[:10])
    expected_vals = [exp.base_values[0] for exp in shap_vals]
    expected_vals = np.array(expected_vals)
    shap_vals_plot = [exp.values[0] for exp in shap_vals]
    shap_vals_plot = np.array(shap_vals_plot)

    print('plotting shap values')

    # Know this works, no need to run it
    # shap.plots.beeswarm(shap_vals, show=False)
    # print('beeswarm complete')

    # shap.decision_plot(expected_vals, shap_vals)
    # print('decision plot complete')

    shap.plots.force(shap_vals)
    print('force plot complete')


