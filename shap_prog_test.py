import joblib

from pathlib import Path
import os

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

    if shap_config.shapefile['type'] == 'points':
        print('Loading point data')
        x_data_point, name_list = uncoverml.shapley.load_data_shap(shap_config, config)
        print('Calculating point shap values')
        shap_vals_point = uncoverml.shapley.calc_shap_vals(model, shap_config, x_data_point)
        # print('Generating point plots')
        # uncoverml.shapley.generate_plots(shap_config.plot_config_list, shap_vals_point, shap_config,
        #                                  name_list=name_list)

        print('Loading point poly data')
        x_data_poly_point, x_poly_coords = uncoverml.shapley.load_point_poly_data(shap_config, config)
        print('Calculating point poly shap values')
        shap_vals_dict = {}
        for name in name_list:
            print(f'Calculating shap values for {name}')
            current_data = x_data_poly_point[name]
            current_shap_vals = uncoverml.shapley.calc_shap_vals(model, shap_config, current_data)
            shap_vals_dict[name] = current_shap_vals

        if shap_config.plot_config_list is not None:
            print('Generating point poly plots')
            uncoverml.shapley.generate_plots_poly_point(name_list, shap_vals_dict, shap_vals_point, shap_config,
                                                        output_names=shap_config.output_names, lon_lats=x_poly_coords)
    else:
        print('Loading poly data')
        x_all, x_coords = uncoverml.shapley.load_data_shap(shap_config, config)
        print('Calculating poly shap values')
        shap_vals = uncoverml.shapley.calc_shap_vals(model, shap_config, x_all)
        if shap_config.do_save:
            save_dict = {
                'shap_vals': shap_vals,
                'x_coords': x_coords
            }
            Path(shap_config.output_path).mkdir(parents=True, exist_ok=True)
            data_save_path = os.path.join(shap_config.output_path, shap_config.save_name + '.data')
            joblib.dump(save_dict, data_save_path)

        if shap_config.plot_config_list is not None:
            print('Generating poly plots')
            uncoverml.shapley.generate_plots(shap_config.plot_config_list, shap_vals, shap_config)

    print('Shap process complete')


if __name__ == '__main__':
    model_config_file = 'gbquantile/gbquantiles.model'
    shap_yaml = '/g/data/ge3/as6887/working-folder/uncover-ml/shap_point_test.yaml'
    shapley_cli(model_config_file, shap_yaml)
