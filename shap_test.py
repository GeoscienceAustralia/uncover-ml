import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt

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

    print('plotting shap values')

    shap.plots.beeswarm(shap_vals, show=False)
    print('beeswarm complete')

    shap.plots.decision(shap_vals)
    print('decision plot complete')

    # shap.plots.force(explainer.expected_value, shap_vals, x_all)
    # print('force plot complete')


