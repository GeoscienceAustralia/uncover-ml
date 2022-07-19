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
    predictions = predict.predict(x_vals, model, interval=config.quantiles, lon_lat=targets_all.positions)
    return predictions


if __name__ == '__main__':
    print('creating explainer')
    explainer = shap.Explainer(predict_for_shap, x_all)
    print('calculating shap values')
    shap_vals = explainer(x_all[:10])
    print('plotting shap values')
    shap.plots.beeswarm(shap_vals, show=False)
    plt.savefig('test.svg')
    print('test shap done')
