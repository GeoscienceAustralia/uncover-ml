import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt

from uncoverml import config, predict
from uncoverml.scripts import uncoverml as uncli

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
    exp = shap.Explanation(shap_vals.values, shap_vals.base_values[0][0], shap_vals.data)
    shap.plots.waterfall(shap_vals[0], show=False)
    plt.savefig('test.svg')
    print('test shap done')
