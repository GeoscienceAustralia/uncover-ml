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
    if shap_config.shapefile['type'] == 'polygon':
        lon_lats = uncoverml.shapley.get_coords_info(shap_config)
        print('Got lons and lats')

    print('data_loaded')

    print('calculating shap values')
    shap_vals = uncoverml.shapley.calc_shap_vals(model, shap_config, x_all)

    print('generating plots')
    uncoverml.shapley.generate_plots(shap_config.plot_config_list, shap_vals, shap_config)


if __name__ == '__main__':
    model_config_file = 'gbquantile/gbquantiles.model'
    shap_yaml = '/g/data/ge3/as6887/working-folder/uncover-ml/shap_polygon_test.yaml'
    shapley_cli(model_config_file, shap_yaml)
