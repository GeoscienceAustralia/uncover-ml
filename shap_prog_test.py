import joblib

import uncoverml.config
import uncoverml.shapley

from uncoverml.scripts import uncoverml as uncli


def shapley_cli(model_file, shapley_yaml):

    with open(model_file, 'rb') as f:
        state_dict = joblib.load(f)

    model = state_dict["model"]
    config = state_dict["config"]

    print('loading config')
    shap_config = uncoverml.shapley.ShapConfig(shapley_yaml, config)

    print('loading data')
    # noinspection PyProtectedMember
    targets_all, x_all = uncli._load_data(config, partitions=200)

    print('calculating shap values')
    shap_vals = uncoverml.shapley.calc_shap_vals(model, shap_config, x_all)

    print('generating plots')
    uncoverml.shapley.generate_plots(shap_config.plot_config_list, shap_vals, shap_config)


if __name__ == '__main__':
    model_file = 'gbquantile/gbquantiles.model'
    shap_yaml = '/g/data/ge3/as6887/working-folder/uncover-ml/shap_test.yaml'
    shapley_cli(model_file, shap_yaml)
