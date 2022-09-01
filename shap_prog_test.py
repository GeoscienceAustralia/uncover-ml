import joblib

from pathlib import Path

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
    x_all = uncoverml.shapley.load_data_shap(shap_config, config)
    print('data_loaded')

    for key, value in x_all.items():
        print(f'Calculating shap values for point {key}')
        shap_vals = uncoverml.shapley.calc_shap_vals(model, shap_config, value)
        print('Calculation complete')

        print(f'Generating plots for point {key}')
        current_point_name = key if shap_config.shapefile['type'] == 'points' else None
        uncoverml.shapley.generate_plots(shap_config.plot_config_list, shap_vals, shap_config,
                                         point_name=current_point_name)


if __name__ == '__main__':
    model_config_file = 'gbquantile/gbquantiles.model'
    shap_yaml = '/g/data/ge3/as6887/working-folder/uncover-ml/shap_point_test.yaml'
    shapley_cli(model_config_file, shap_yaml)
