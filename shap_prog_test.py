import joblib

import uncoverml.config
import uncoverml.shapley

from uncoverml.scripts import uncoverml as uncli


def shapley_cli(model_file, shapley_yaml):

    with open(model_file, 'rb') as f:
        state_dict = joblib.load(f)

    model = state_dict["model"]
    config = state_dict["config"]

    shap_config = uncoverml.shapley.ShapConfig(shapley_yaml)
    # noinspection PyProtectedMember
    targets_all, x_all = uncli._load_data(config, partitions=200)

    shap_vals = uncoverml.shapley.calc_shap_vals(model, shap_config, x_all)
    uncoverml.shapley.generate_plots(shap_config.plot_config_list, shap_vals)
