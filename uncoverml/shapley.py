import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import logging
from os import path
from os import makedirs
from pathlib import Path
import glob
import csv
import yaml
from pathlib import Path

from uncoverml import predict

log = logging.getLogger(__name__)

'''
Properties for shap config
- Explainer
- Explainer kwargs
- Calc start row
- Calc end row
- Masker
    - Type
    - Start row
    - End row
    - Kwargs
    - Mask list
- Output path
- Plots each individually

'''


class ShapConfig:
    # REMEMBER, DATA AND FEATURE NAMES CAN BE ACCESSED FROM THE MAIN CONFIG
    def __init__(self, yaml_file, main_config):
        with open(yaml_file, 'r') as f:
            s = yaml.safe_load(f)
        self.name = path.basename(yaml_file).rsplit(".", 1)[0]

        if 'explainer' in s:
            self.explainer = s['explainer']
        else:
            self.explainer = None
            log.error('No explainer provided, cannot calculate Shapley values')

        self.explainer_kwargs = s['explainer_kwargs'] if 'explainer_kwargs' in s else None
        self.calc_start_row = s['calc_start_row'] if 'calc_start_row' in s else None
        self.calc_end_row = s['calc_end_row'] if 'calc_end_row' in s else None
        self.masker = s['masker'] if 'masker' in s else None

        if 'output_path' in s:
            self.output_path = s['output_path']
        elif hasattr(main_config, 'output_dir'):
            self.output_path = path.join(main_config.output_dir, 'shap')
        else:
            self.output_path = None
            logging.error('No output path, exception will be thrown when saving')

        self.plot_config_list = None
        if 'plots' in s:
            plot_configs = s['plots']
            self.plot_config_list = [PlotConfig(p) for p in plot_configs]
        else:
            log.warning('No plots will be created')


explainer_map = {
    'explainer': {'function': shap.Explainer,
                  'requirements': ['masker'],
                  'allowed': ['independent', 'partition', 'data', 'list']}
}

masker_map = {
    'independent': shap.maskers.independent,
    'partition': shap.maskers.partition
}


def select_masker(mask_info, mask_data):
    # Might nest this into prepare_check_masker later
    masker = None
    mask_type = mask_info['type']
    if mask_type in ['independent', 'partition']:
        if 'kwargs' in mask_info:
            masker = masker_map[mask_type](mask_data, **mask_info['kwargs'])
        else:
            masker = masker_map[mask_type](mask_data)
    elif mask_type == 'data':
        masker = mask_data

    return masker


def prepare_check_masker(shap_config, x_data):
    if shap_config.masker['type'] not in explainer_map[shap_config.explainer]['allowed']:
        log.error('Incompatible masker specified')
        return None

    mask_info = shap_config.masker
    start_row = mask_info['start_row'] if 'start_row' in mask_info else 0
    end_row = mask_info['end_row'] if 'end_row' in mask_info else -1
    mask_data = x_data[start_row:end_row]
    if mask_info['type'] == 'list':
        if 'mask_list' not in mask_info:
            log.error('Mask list required, but not defined')
            return None

        if len(mask_info['mask_list']) != mask_data.shape[1]:
            log.error('Number of maskers in list must match number of features')
            return None

        mask_var = [select_masker(type, mask_data) for type in mask_info['mask_list']]
    else:
        mask_var = select_masker(mask_info['type'], mask_data)

    return mask_var


def gather_explainer_req(shap_config):
    model_req = explainer_map[shap_config.explainer]['requirements']
    requirements = []
    reqs_fulfilled = 0
    for req in model_req:
        # Can expand this loop for more requirements later
        if req == 'masker':
            if shap_config.masker is not None:
                mask_var = prepare_check_masker(shap_config, x_data)
                if (mask_var is None) or (None in mask_var):
                    logging.error('Cannot proceed, there are undefined maskers')
                    return None

                reqs_fulfilled += 1
                requirements.append(mask_var)
            else:
                log.error('Masker requirement not found in config')
                return None

    ret_val = (requirements, reqs_fulfilled)
    return ret_val


def calc_shap_vals(model, shap_config, x_data):
    def shap_predict(x):
        pred_vals = predict.predict(x, model)
        return pred_vals

    if (shap_config.explainer not in explainer_map) or (shap_config.explainer is None):
        log.error('Invalid or no explainer specified')
        return None

    explainer_reqs = gather_explainer_req(shap_config)
    if explainer_reqs is None:
        log.error('Explainer requirements did not come out correctly')
        return None

    reqs, reqs_fulfilled = explainer_reqs
    reqs = tuple(reqs)
    if reqs_fulfilled < len(x_data):
        logging.warning('Some explainer requirements not fulfilled, calculation might not work')

    if shap_config.explainer_kwargs is not None:
        explainer_obj = explainer_map[shap_config.explainer](shap_predict, *reqs, **shap_config.explainer_kwargs)
    else:
        explainer_obj = explainer_map[shap_config.explainer](shap_predict, *reqs)

    calc_start_row = shap_config.calc_start_row if shap_config.calc_start_row is not None else 0
    calc_end_row = shap_config.calc_end_row if shap_config.calc_end_row is not None else -1
    calc_data = x_data[calc_start_row:calc_end_row]
    shap_vals = explainer_obj(calc_data)
    return shap_vals


class PlotConfig:
    def __init__(self, plot_dict):
        if 'plot_name' in plot_dict:
            self.plot_name = plot_dict['plot_name']
        else:
            logging.error('Plot name is need to uniquely identify plots')

        if 'type' in plot_dict:
            self.type = plot_dict['type']
        else:
            logging.error('Need to specify a plot type')

        self.plot_title = plot_dict['plot_title'] if 'plot_title' in plot_dict else None
        self.output_idx = plot_dict['output_idx'] if 'output_idx' in plot_dict else None
        self.plot_features = plot_dict['plot_features'] if 'plot_features' in plot_dict else None
        self.xlim = plot_dict['xlim'] if 'xlim' in plot_dict else None
        self.ylim = plot_dict['ylim'] if 'ylim' in plot_dict else None


'''
Types of plot:
    - Summary
    - Bar
    - Decision
    - Shap Correlation
    
    - Spatial
    - Scatter
'''


def save_plot(fig, plot_name, shap_config):
    plot_save_path = path.join(shap_config.output_path, 'shap', plot_name + '.png')
    fig.savefig(plot_save_path)


def aggregate_subplot(plot_vals, plot_config, shap_config, **kwargs):
    num_plots = plot_vals.shape[2] if len(plot_vals.shape) > 2 else 1
    fig, axs = plt.subplots(1, num_plots)
    for idx in range(num_plots):
        current_plot_data = plot_vals[:, :, idx] if num_plots > 1 else plot_vals
        plotting_func_map[plot_config.type](current_plot_data, plot_config, axs[idx], **kwargs)

    if plot_config.name is not None:
        plot_name = plot_config.name
    else:
        plot_name = plot_config.type

    save_plot(fig, plot_name, shap_config)
    plt.clf()


def aggregate_separate(plot_vals, plot_config, **kwargs):
    num_plots = plot_vals.shape[2] if len(plot_vals.shape) > 2 else 1
    fig, ax = plt.subplots()
    for idx in range(num_plots):
        current_plot_data = plot_vals[:, :, idx] if num_plots > 1 else plot_vals
        plotting_func_map[plot_config.type](current_plot_data, plot_config, ax, **kwargs)

        if plot_config.name is not None:
            plot_name = f'{plot_config.name}_{idx}'
        else:
            plot_name = f'{plot_config.type}_{idx}'

        save_plot(fig, plot_name, shap_config)
        plt.clf()


def summary_plot(plot_data, plot_config, target_ax, **kwargs):
    feature_names = kwargs['feature_names'] if 'feature_names' in kwargs else None
    plt.sca(target_ax)
    shap.summary_plot(plot_data.values, features=plot_data.data, feature_names=feature_names, show=False)


def bar_plot(plot_data, plot_config, target_ax, **kwargs):
    feature_names = kwargs['feature_names'] if 'feature_names' in kwargs else None
    plt.sca(target_ax)
    shap.bar_plot(plot_data.values, features=plot_data.data, feature_names=feature_names, show=False)


def shap_corr_plot(plot_data, plot_config, target_ax, **kwargs):
    if 'feature_names' in kwargs:
        feature_names = kwargs['feature_names']
    else:
        feature_names = [str(x) for x in range(plot_data.shape[1])]

    plot_dataframe = pd.DataFrame(plot_data.values, columns=feature_names)
    corr_matrix = plot_dataframe.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', fmt='.1g', annot=False, ax=target_ax)


def decision_plot(plot_data, plot_config, target_ax, **kwargs):
    feature_names = kwargs['feature_names'] if 'feature_names' in kwargs else None
    plt.sca(target_ax)
    shap.decision_plot(plot_data.base_value[0], plot_data.values, feature_names=feature_names)


def spatial_plots(shap_vals, plot_config, shap_config, **kwargs):
    if 'lon_lat' not in kwargs:
        log.error('No lon-lat info, cannot plot')
        return None

    if 'feature_names' in kwargs:
        feature_names = kwargs['feature_names']
    else:
        feature_names = [str(x) for x in range(shap_vals.shape[1])]

    multi_output_dim = shap_vals.shape[2] if len(shap_vals.shape) else 1
    fig, ax = plt.subplots()
    cm = plt.cm.get_cmap('cool')
    lon_lat = kwargs['lon_lat']
    for dim_idx in range(multi_output_dim):
        for feat_idx in range(len(feature_names)):
            plot_vals = shap_vals[:, feat_idx, dim_idx]
            current_plot = ax.scatter(lon_lat[:, 0], lon_lat[:, 1], s=10, c=plot_vals, cmap=cm)
            fig.colorbar(current_plot)

            if plot_config.name is not None:
                plot_name = plot_config.name
            else:
                plot_name = plot_config.type

            save_plot(fig, plot_name, shap_config)
            fig.clf()


def scatter_plot(shap_vals, plot_config, shap_config, **kwargs):
    if 'feature_names' in kwargs:
        feature_names = kwargs['feature_names']
    else:
        feature_names = [str(x) for x in range(shap_vals.shape[1])]

    if plot_config.plot_features is None:
        log.error('No plot features provided, cannot plot')
        return None

    fig, ax = plt.subplots()
    multi_output_dim = shap_vals.shape[2] if len(shap_vals.shape) else 1
    for dim_idx in range(multi_output_dim):
        for feat in plot_config.plot_features:
            inter_feat = feat[1] if len(feat) == 2 else 'auto'
            plot_vals = shap_vals[:, :, dim_idx]
            shap.dependence_plot(feat[0], plot_vals.values, plot_vals.data, feature_names=feature_names,
                                 interaction_index=inter_feat, show=False, ax=ax)

            if plot_config.name is not None:
                plot_name = plot_config.name
            else:
                plot_name = plot_config.type

            save_plot(fig, plot_name, shap_config)
            fig.clf()


plotting_func_map = {
    'summary': summary_plot,
    'bar': bar_plot,
    'decision': decision_plot,
    'shap_corr': shap_corr_plot
}

plotting_type_map = {
    'summary': aggregate_subplot,
    'bar': aggregate_subplot,
    'decision': aggregate_separate,
    'shap_corr': aggregate_separate,
    'spatial': spatial_plot,
    'scatter': scatter_plot
}


def generate_plots(plot_config_list, shap_vals, **kwargs):
    if 'feature_names' not in kwargs:
        log.warning('Feature names not provided, plots might be confusing')

    for current_plot_config in plot_config_list:
        plot_vals = shap_vals
        if current_plot_config.output_idx is not None:
            plot_vals = shap_vals[:, :, current_plot_config.output_idx]

        plotting_type_map[current_plot_config.type](plot_vals, current_plot_config, **kwargs)


