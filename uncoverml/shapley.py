import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import shapefile

import logging
from os import path
from os import makedirs
from pathlib import Path
import glob
import csv
import yaml
from pathlib import Path
from collections.abc import Iterable

from uncoverml import predict
from uncoverml import mpiops
from uncoverml import geoio
from uncoverml import features
from uncoverml import image
from uncoverml import patch

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


def get_shapefile_lon_lat(file_to_load):
    sf = shapefile.Reader(file_to_load)
    # Get coordinates
    coords = []
    for shape in sf.iterShapes():
        coords.append(list(shape.__geo_interface__['coordinates']))
    label_coords = np.array(coords).squeeze()
    return label_coords


def patch_at_coords(image, patchsize, coords):
    # noinspection PyProtectedMember
    data, mask, data_dtype = patch._image_to_data(image)

    valid = image.in_bounds(coords)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) > 0:
        valid_lonlats = lonlats[valid]
        pixels = image.lonlat2pix(valid_lonlats)
        patches = patch.point_patches(data, patchsize, pixels)
        patch_mask = patch.point_patches(mask, patchsize, pixels)
        patch_array = np.ma.masked_array(data=patches, mask=patch_mask)

    else:
        patchwidth = 2 * patchsize + 1
        shp = (0, patchwidth, patchwidth, data.shape[2])
        patch_data = np.zeros(shp, dtype=data_dtype)
        patch_mask = np.zeros(shp, dtype=bool)
        patch_array = np.ma.masked_array(data=patch_data, mask=patch_mask)

    return patch_array


def _extract_from_chunk(image_source, lon_lat, chunk_index, total_chunks,
                        patchsize):
    image_chunk = image.Image(image_source, chunk_index, total_chunks, patchsize)
    # figure out which chunks I need to consider
    y_min = lon_lat[0, 1]
    y_max = lon_lat[-1, 1]
    # noinspection PyProtectedMember
    if features._image_has_targets(y_min, y_max, image_chunk):
        x = patch_at_coords(image_chunk, patchsize, lon_lat)
    else:
        x = None
    return x


def extract_features_shap(image_source, lon_lat, n_subchunks, patchsize):
    equiv_chunks = n_subchunks * mpiops.chunks
    x_all = []
    for i in range(equiv_chunks):
        x = _extract_from_chunk(image_source, lon_lat, i, equiv_chunks,
                                patchsize)
        if x is not None:
            x_all.append(x)
    if len(x_all) > 0:
        x_all_data = np.concatenate([a.data for a in x_all], axis=0)
        x_all_mask = np.concatenate([a.mask for a in x_all], axis=0)
        x_all = np.ma.masked_array(x_all_data, mask=x_all_mask)
    else:
        raise ValueError("All shapefile locations lie outside image boundaries")
    assert x_all.shape[0] == lon_lat.shape[0]
    return x_all


def image_feature_sets_shap(lon_lat, main_config):
    def get_data(image_source):
        r = extract_features_shap(image_source, lon_lat, main_config.n_subchunks, main_config.patchsize)
        return r

    # noinspection PyProtectedMember
    result = geoio._iterate_sources(get_data, main_config)
    return result


def load_data_shap(calc_shapefile, main_config):
    if mpiops.chunk_index == 0:
        lonlat = get_shapefile_lon_lat(calc_shapefile)
        ordind = np.lexsort(lonlat.T)
        lonlat = lonlat[ordind]
        lonlat = np.array_split(lonlat, mpiops.chunks)
    else:
        lonlat = None

    image_chunk_sets = image_feature_sets_shap(lonlat, main_config)
    transform_sets = [k.transform_set for k in config.feature_sets]
    transformed_features, keep = ls.features.transform_features(image_chunk_sets,
                                                    transform_sets,
                                                    main_config.final_transform,
                                                    main_config)
    x_all = ls.features.gather_features(transformed_features[keep], node=0)
    return x_all


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
    'independent': shap.maskers.Independent,
    'partition': shap.maskers.Partition
}


def select_masker(mask_type, mask_data, mask_info=None):
    # Might nest this into prepare_check_masker later
    masker = None
    if mask_type in ['independent', 'partition']:
        if (mask_info is not None) and ('kwargs' in mask_info):
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

        # Might be a good idea to pass mask_info here - future improvement
        mask_var = [select_masker(type, mask_data) for type in mask_info['mask_list']]
    else:
        mask_var = select_masker(mask_info['type'], mask_data, mask_info)

    return mask_var


def gather_explainer_req(shap_config, x_data):
    model_req = explainer_map[shap_config.explainer]['requirements']
    requirements = []
    reqs_fulfilled = len(model_req)
    for req in model_req:
        # Can expand this loop for more requirements later
        if req == 'masker':
            if shap_config.masker is not None:
                mask_var = prepare_check_masker(shap_config, x_data)
                if mask_var is None:
                    logging.error('Cannot proceed, there are undefined maskers')
                    return None

                if isinstance(mask_var, Iterable) and (None in mask_var):
                    logging.error('Cannot proceed, there are undefined maskers')
                    return None

                reqs_fulfilled -= 1
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

    explainer_reqs = gather_explainer_req(shap_config, x_data)
    if explainer_reqs is None:
        log.error('Explainer requirements did not come out correctly')
        return None

    reqs, reqs_fulfilled = explainer_reqs
    reqs = tuple(reqs)
    if reqs_fulfilled > 0:
        logging.warning('Some explainer requirements not fulfilled, calculation might not work')

    if shap_config.explainer_kwargs is not None:
        explainer_obj = explainer_map[shap_config.explainer]['function'](shap_predict, *reqs, **shap_config.explainer_kwargs)
    else:
        explainer_obj = explainer_map[shap_config.explainer]['function'](shap_predict, *reqs)

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
    Path(shap_config.output_path).mkdir(parents=True, exist_ok=True)
    plot_save_path = path.join(shap_config.output_path, plot_name + '.png')
    fig.savefig(plot_save_path, dpi=500)


def aggregate_subplot(plot_vals, plot_config, shap_config, **kwargs):
    num_plots = plot_vals.shape[2] if len(plot_vals.shape) > 2 else 1
    fig, axs = plt.subplots(1, num_plots, figsize=(16.53, 11.69))
    for idx in range(num_plots):
        current_plot_data = plot_vals[:, :, idx] if num_plots > 1 else plot_vals
        plotting_func_map[plot_config.type](current_plot_data, plot_config, axs[idx], idx, **kwargs)

    if plot_config.plot_name is not None:
        plot_name = plot_config.plot_name
    else:
        plot_name = plot_config.type

    plt.tight_layout()
    save_plot(fig, plot_name, shap_config)
    plt.clf()


def aggregate_separate(plot_vals, plot_config, shap_config, **kwargs):
    num_plots = plot_vals.shape[2] if len(plot_vals.shape) > 2 else 1
    fig, ax = plt.subplots(figsize=(32, 22), sharey=True)
    for idx in range(num_plots):
        current_plot_data = plot_vals[:, :, idx] if num_plots > 1 else plot_vals
        plotting_func_map[plot_config.type](current_plot_data, plot_config, ax, idx, **kwargs)

        if plot_config.plot_name is not None:
            plot_name = f'{plot_config.plot_name}_{idx}'
        else:
            plot_name = f'{plot_config.type}_{idx}'

        plt.tight_layout()
        save_plot(fig, plot_name, shap_config)
        plt.clf()


def summary_plot(plot_data, plot_config, target_ax, plot_idx, **kwargs):
    feature_names = kwargs['feature_names'] if 'feature_names' in kwargs else None
    plt.sca(target_ax)
    shap.summary_plot(plot_data.values, features=plot_data.data, feature_names=feature_names,
                      max_display=plot_data.shape[1], sort=False, show=False)

    if plot_idx > 0:
        target_ax.axes.yaxis.set_visible(False)
        x_axis = target_ax.axes.get_xaxis()
        x_label = x_axis.get_label()
        x_label.set_visible(False)

    plt.gcf().axes[-1].remove()


def bar_plot(plot_data, plot_config, target_ax, idx, **kwargs):
    feature_names = kwargs['feature_names'] if 'feature_names' in kwargs else None
    plt.sca(target_ax)
    shap.plots.bar(plot_data, show=False)


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
    # plt.sca(target_ax)
    shap.decision_plot(plot_data.base_values[0], plot_data.values, feature_names=feature_names)


def spatial_plot(shap_vals, plot_config, shap_config, **kwargs):
    if 'lon_lat' not in kwargs:
        log.error('No lon-lat info, cannot plot')
        return None

    if 'feature_names' in kwargs:
        feature_names = kwargs['feature_names']
    else:
        feature_names = [str(x) for x in range(shap_vals.shape[1])]

    multi_output_dim = shap_vals.shape[2] if len(shap_vals.shape) else 1
    fig, ax = plt.subplots(figsize=(16.53, 11.69))
    cm = plt.cm.get_cmap('cool')
    lon_lat = kwargs['lon_lat']
    for dim_idx in range(multi_output_dim):
        for feat_idx in range(len(feature_names)):
            plot_vals = shap_vals[:, feat_idx, dim_idx]
            current_plot = ax.scatter(lon_lat[:, 0], lon_lat[:, 1], s=10, c=plot_vals, cmap=cm)
            fig.colorbar(current_plot)

            if plot_config.plot_name is not None:
                plot_name = plot_config.plot_name
            else:
                plot_name = plot_config.type

            plt.tight_layout()
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

    fig, ax = plt.subplots(figsize=(16.53, 11.69))
    multi_output_dim = shap_vals.shape[2] if len(shap_vals.shape) else 1
    for dim_idx in range(multi_output_dim):
        for feat in plot_config.plot_features:
            inter_feat = feat[1] if len(feat) == 2 else 'auto'
            plot_vals = shap_vals[:, :, dim_idx]
            shap.dependence_plot(feat[0], plot_vals.values, plot_vals.data, feature_names=feature_names,
                                 interaction_index=inter_feat, show=False, ax=ax)

            if plot_config.plot_name is not None:
                plot_name = plot_config.plot_name
            else:
                plot_name = plot_config.type

            plt.tight_layout()
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


def generate_plots(plot_config_list, shap_vals, shap_config, **kwargs):
    if 'feature_names' not in kwargs:
        log.warning('Feature names not provided, plots might be confusing')

    for current_plot_config in plot_config_list:
        plot_vals = shap_vals
        if current_plot_config.output_idx is not None:
            plot_vals = shap_vals[:, :, current_plot_config.output_idx]

        plotting_type_map[current_plot_config.type](plot_vals, current_plot_config, shap_config, **kwargs)


