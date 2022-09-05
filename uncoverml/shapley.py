import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import seaborn as sns
import geopandas as gpd
import shapefile
import rasterio
from rasterio.mask import mask
from shapely import geometry
from shapely.geometry import mapping, MultiPolygon, Point
from shapely.ops import transform
from functools import partial
from rasterio import Affine
import pyproj
from textwrap import wrap

from multiprocessing import Pool

import logging
from os import path
from os import makedirs
from pathlib import Path
import glob
import csv
import yaml
from pathlib import Path
from collections.abc import Iterable
from collections import OrderedDict

from uncoverml import predict
from uncoverml import mpiops
from uncoverml import geoio
from uncoverml import features
from uncoverml import image
from uncoverml import patch
from uncoverml.transforms.transformset import missing_percentage

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


def intersect_shp(single_row_df, image_source_dir, **kwargs):
    geoms = single_row_df.geometry.values[0]
    geoms = [mapping(geoms)]
    # extract the raster values within the polygon
    with rasterio.open(image_source_dir) as src:
        out_image, out_transform = mask(src, geoms, crop=True)

    return out_image, out_transform


def get_data_points(loaded_shapefile, image_source):
    res_list = []
    for idx, row in loaded_shapefile.iterrows():
        single_row_df = loaded_shapefile.iloc[[idx]]
        (result, transform) = intersect_shp(single_row_df, image_source)
        res_list.append(result)

    return np.concatenate(res_list)


def get_data_polygon(loaded_shapefile, image_source):
    (result, transform) = intersect_shp(loaded_shapefile, image_source)
    return result


def image_feature_sets_shap(shap_config, main_config):
    loaded_shapefile = gpd.read_file(shap_config.shapefile['dir'])
    name_list = None
    if shap_config.shapefile['type'] == 'points':
        name_list = loaded_shapefile['Name'].to_list()

    results = []
    for s in main_config.feature_sets:
        extracted_chunks = {}
        for tif in s.files:
            print(tif)
            name = path.abspath(tif)
            if shap_config.shapefile['type'] == 'points':
                x = get_data_points(loaded_shapefile, name)
            else:
                x = get_data_polygon(loaded_shapefile, name)

            val_count = x.size
            x = np.reshape(x, (val_count, 1, 1, 1))
            x = ma.array(x, mask=np.zeros([val_count, 1, 1, 1]))
            # TODO this may hurt performance. Consider removal
            if type(x) is np.ma.MaskedArray:
                count = mpiops.count(x)
                # if not np.all(count > 0):
                #     s = ("{} has no data in at least one band.".format(name) +
                #          " Valid_pixel_count: {}".format(count))
                #     raise ValueError(s)
                missing_percent = missing_percentage(x)
                t_missing = mpiops.comm.allreduce(
                    missing_percent) / mpiops.chunks
                log.info("{}: {}px {:2.2f}% missing".format(
                    name, count, t_missing))
            extracted_chunks[name] = x
        extracted_chunks = OrderedDict(sorted(
            extracted_chunks.items(), key=lambda t: t[0]))

        results.append(extracted_chunks)

    if shap_config.shapefile['type'] == 'points':
        return results, name_list
    else:
        return results


def load_data_shap(shap_config, main_config):
    image_chunk_sets = image_feature_sets_shap(shap_config, main_config)
    name_list = None
    if shap_config.shapefile['type'] == 'points':
        image_chunk_sets, name_list = image_chunk_sets

    transform_sets = [k.transform_set for k in main_config.feature_sets]
    transformed_features, keep = features.transform_features(image_chunk_sets,
                                                    transform_sets,
                                                    main_config.final_transform,
                                                    main_config)
    x_all = features.gather_features(transformed_features[keep], node=0)

    if shap_config.shapefile['type'] == 'points':
        return x_all, name_list
    else:
        return x_all


def load_point_poly_data(shap_config, main_config):
    loaded_shapefile = gpd.read_file(shap_config.shapefile['dir'])
    name_list = loaded_shapefile['Name'].to_list()

    out_result = {}
    for name in name_list:
        print(f'Getting data for {name}')
        current_row = loaded_shapefile[loaded_shapefile['Name'] == name]
        current_poly_data = gen_poly_data(current_row, shap_config, main_config)
        out_result[name] = current_poly_data

    return out_result


def gen_poly_data(single_row_df, shap_config, main_config):
    size = shap_config.shapefile['size']
    image_chunk_sets = gen_poly_from_point(single_row_df, main_config, size)
    transform_sets = [k.transform_set for k in main_config.feature_sets]
    transformed_features, keep = features.transform_features(image_chunk_sets,
                                                             transform_sets,
                                                             main_config.final_transform,
                                                             main_config)
    x_all = features.gather_features(transformed_features[keep], node=0)
    return x_all


def gen_poly_from_point(single_row_df, main_config, size):
    results = []
    for s in main_config.feature_sets:
        extracted_chunks = {}
        for tif in s.files:
            name = path.abspath(tif)
            x = intersect_point_neighbourhood(single_row_df, size, name)
            val_count = x.size
            # print(f'{tif}: {val_count}')
            x = np.reshape(x, (val_count, 1, 1, 1))
            x = ma.array(x, mask=np.zeros([val_count, 1, 1, 1]))
            # TODO this may hurt performance. Consider removal
            if type(x) is np.ma.MaskedArray:
                count = mpiops.count(x)
                # if not np.all(count > 0):
                #     s = ("{} has no data in at least one band.".format(name) +
                #          " Valid_pixel_count: {}".format(count))
                #     raise ValueError(s)
                missing_percent = missing_percentage(x)
                t_missing = mpiops.comm.allreduce(
                    missing_percent) / mpiops.chunks
                log.info("{}: {}px {:2.2f}% missing".format(
                    name, count, t_missing))
            extracted_chunks[name] = x
        extracted_chunks = OrderedDict(sorted(
            extracted_chunks.items(), key=lambda t: t[0]))

        results.append(extracted_chunks)

    return results


def intersect_point_neighbourhood(single_row_df, size, image_source_dir):
    single_point = single_row_df.geometry.values[0]
    with rasterio.open(image_source_dir) as src:
        py, px = src.index(single_point.x, single_point.y)
        window = rasterio.windows.Window(px - size // 2, py - size // 2, size, size)
        out_image = src.read(window=window)

    return out_image


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

        if 'shapefile' in s:
            self.shapefile = s['shapefile']
        else:
            self.shapefile = None
            log.error('No shapefile provided, calculation will fail')

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

        self.output_names = s['output_names'] if 'output_names' in s else None
        self.feature_path = s['feature_path'] if 'feature_path' in s else None


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


def calc_shap_vals(model, shap_config, x_data, num_proc=1):
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

    if shap_config.calc_start_row is not None:
        x_data = x_data[shap_config.calc_start_row:]

    if shap_config.calc_end_row is not None:
        x_data = x_data[:shap_config.calc_end_row]

    calc_data = x_data
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
    plt.tight_layout(pad=3)
    fig.savefig(plot_save_path, dpi=100)


common_x_text_map = {
    'summary': 'SHAP value (impact on model output)',
    'bar': 'mean(|SHAP value|) (average impact on model output magnitude)'
}


def aggregate_subplot(plot_vals, plot_config, shap_config, **kwargs):
    num_plots = plot_vals.shape[2] if len(plot_vals.shape) > 2 else 1
    row_height = 0.4
    plot_height = (plot_vals.shape[1] * row_height) + 1.5
    plot_width = 20 if plot_config.type == 'bar' else 16
    fig, axs = plt.subplots(1, num_plots, dpi=100)
    for idx in range(num_plots):
        current_plot_data = plot_vals[:, :, idx] if num_plots > 1 else plot_vals
        plotting_func_map[plot_config.type](current_plot_data, plot_config, axs[idx], idx, **kwargs)

    fig.set_size_inches(plot_width, plot_height, forward=True)
    if plot_config.plot_name is not None:
        plot_name = plot_config.plot_name
    else:
        plot_name = plot_config.type

    common_x_text = common_x_text_map[plot_config.type]
    fig.text(0.5, 0.01, common_x_text, ha='center')
    save_plot(fig, plot_name, shap_config)
    plt.clf()


def individual_subplot(plot_vals, plot_config, shap_config, **kwargs):
    num_plots = plot_vals.shape[0]
    output_dims = plot_vals.shape[2] if len(plot_vals.shape) > 2 else 1

    # 4 plots per figure, 2x2 grid
    groups_of_four = divmod(num_plots, 4)
    num_iter = groups_of_four[0] if groups_of_four[1] == 0 else (groups_of_four[0] + 1)
    start_idx_list = [i * 4 for i in range(num_iter)]
    for output_idx in range(output_dims):
        for start_idx, fig_idx in zip(start_idx_list, list(range(num_iter))):
            max_idx = plot_vals[:, :, output_idx].shape[0] - 1
            end_idx = (start_idx + 3) if (start_idx + 3) < max_idx else max_idx
            current_subplots_vals = plot_vals[start_idx:end_idx, :, output_idx]
            current_subplots_vals.data = current_subplots_vals.data[start_idx:end_idx, :]
            num_plots_current_fig = current_subplots_vals.shape[0]
            nrow = 2 if num_plots_current_fig in [2, 3, 4] else 1
            ncol = 2 if num_plots_current_fig in [3, 4] else 1

            fig, axs = plt.subplots(nrow, ncol, figsize=(1.920, 1.080), dpi=100)
            for val_idx in range(num_plots_current_fig):
                current_plot_data = current_subplots_vals[val_idx]
                current_plot_data.data = current_plot_data.data[val_idx, :]
                val_num = start_idx + val_idx + 1
                plotting_func_map[plot_config.type](current_plot_data, plot_config, np.ravel(axs)[val_idx], val_idx,
                                                    **kwargs, subplot_idx=val_num)

            if plot_config.plot_name is not None:
                plot_name = f'{plot_config.plot_name}_output_{output_idx}_value_{fig_idx}'
            else:
                plot_name = f'{plot_config.type}_output_{output_idx}_value_{fig_idx}'

            save_plot(fig, plot_name, shap_config)
            plt.clf()


def aggregate_separate(plot_vals, plot_config, shap_config, **kwargs):
    num_plots = plot_vals.shape[2] if len(plot_vals.shape) > 2 else 1
    fig, ax = plt.subplots(figsize=(1.920, 1.080), dpi=100, sharey=True)
    for idx in range(num_plots):
        current_plot_data = plot_vals[:, :, idx] if num_plots > 1 else plot_vals
        plotting_func_map[plot_config.type](current_plot_data, plot_config, ax, idx, **kwargs)

        if plot_config.plot_name is not None:
            plot_name = f'{plot_config.plot_name}_{idx}'
        else:
            plot_name = f'{plot_config.type}_{idx}'

        save_plot(fig, plot_name, shap_config)
        plt.clf()


# def force_plot(plot_data, plot_config, target_ax, plot_idx, **kwargs):
#     plt.sca(target_ax)
#     shap.force_plot(plot_data.base_values, shap_values=plot_data.values, features=plot_data.data,
#                     feature_names=plot_data.feature_names, show=False, matplotlib=True)
#     if plot_config.plot_title is not None:
#         current_plot_title = plot_config.plot_title
#         if 'subplot_idx' in kwargs:
#             current_plot_title = current_plot_title + '_' + str(kwargs['subplot_idx'])
#
#         target_ax.title.set_text(current_plot_title)
#         target_ax.tick_params(axis='both', labelsize=5)


def waterfall_plot(plot_data, plot_config, target_ax, plot_idx, **kwargs):
    plt.sca(target_ax)
    shap.waterfall_plot(plot_data)
    if plot_config.plot_title is not None:
        current_plot_title = plot_config.plot_title
        if 'subplot_idx' in kwargs:
            current_plot_title = current_plot_title + '_' + str(kwargs['subplot_idx'])

        if 'point_name' in kwargs:
            current_plot_title = kwargs['point_name'] + '_' + str(kwargs['subplot_idx'])

        target_ax.title.set_text(current_plot_title)
        target_ax.tick_params(axis='both', labelsize=5)


def summary_plot(plot_data, plot_config, target_ax, plot_idx, **kwargs):
    plt.sca(target_ax)
    row_height = 0.4
    plot_height = (plot_data.shape[1] * row_height) + 1.5
    plot_width = 16
    shap.summary_plot(plot_data.values, features=plot_data.data, feature_names=plot_data.feature_names, show=False,
                      max_display=plot_data.shape[1], sort=False, plot_size = (plot_width, plot_height))

    x_axis = target_ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    if plot_idx > 0:
        target_ax.axes.yaxis.set_visible(False)

    plt.gcf().axes[-1].remove()
    if plot_config.plot_title is not None:
        current_plot_title = plot_config.plot_title
        if 'output_idx' in kwargs:
            current_plot_title = current_plot_title + '_' + str(kwargs['output_idx'])
        else:
            current_plot_title = f'{current_plot_title}_{plot_idx}'

        target_ax.title.set_text(current_plot_title)


def bar_plot(plot_data, plot_config, target_ax, plot_idx, **kwargs):
    plt.sca(target_ax)
    shap.plots.bar(plot_data, show=False, max_display=plot_data.shape[1])
    # target_ax.tick_params(axis='both', labelsize=5)
    x_axis = target_ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)

    if plot_config.plot_title is not None:
        current_plot_title = plot_config.plot_title
        if 'output_idx' in kwargs:
            current_plot_title = current_plot_title + '_' + str(kwargs['output_idx'])
        else:
            current_plot_title = f'{current_plot_title}_{plot_idx}'

        target_ax.title.set_text(current_plot_title)


def shap_corr_plot(plot_data, plot_config, target_ax, **kwargs):
    if 'feature_names' in kwargs:
        feature_names = plot_data.feature_names
    else:
        feature_names = [str(x) for x in range(plot_data.shape[1])]

    plot_dataframe = pd.DataFrame(plot_data.values, columns=feature_names)
    corr_matrix = plot_dataframe.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', fmt='.1g', annot=False, ax=target_ax)
    target_ax.tick_params(axis='both', labelsize=5)
    if plot_config.plot_title is not None:
        current_plot_title = plot_config.plot_title
        if 'output_idx' in kwargs:
            current_plot_title = current_plot_title + '_' + str(kwargs['output_idx'])

        target_ax.title.set_text(current_plot_title)


def decision_plot(plot_data, plot_config, target_ax, **kwargs):
    # plt.sca(target_ax)
    shap.decision_plot(plot_data.base_values[0], plot_data.values, feature_names=plot_data.feature_names)
    target_ax.tick_params(axis='both', labelsize=5)
    if plot_config.plot_title is not None:
        current_plot_title = plot_config.plot_title
        if 'output_idx' in kwargs:
            current_plot_title = current_plot_title + '_' + str(kwargs['output_idx'])

        target_ax.title.set_text(current_plot_title)


def spatial_plot(shap_vals, plot_config, shap_config, **kwargs):
    if 'lon_lat' not in kwargs:
        log.error('No lon-lat info, cannot plot')
        return None

    if 'feature_names' in kwargs:
        feature_names = shap_vals.feature_names
    else:
        feature_names = [str(x) for x in range(shap_vals.shape[1])]

    multi_output_dim = shap_vals.shape[2] if len(shap_vals.shape) > 2 else 1
    fig, ax = plt.subplots(figsize=(1.920, 1.080), dpi=100)
    cm = plt.cm.get_cmap('cool')
    lon_lat = kwargs['lon_lat']
    for dim_idx in range(multi_output_dim):
        for feat_idx in range(len(feature_names)):
            plot_vals = shap_vals[:, feat_idx, dim_idx]
            current_plot = ax.scatter(lon_lat[:, 0], lon_lat[:, 1], s=10, c=plot_vals, cmap=cm)
            fig.colorbar(current_plot)

            current_plot_title = plot_config.plot_title if plot_config.plot_title is not None else ''
            add_on_string = f'_output_{dim_idx}_feature_{feature_names[feat_idx]}'
            current_plot_title = current_plot_title + add_on_string
            ax.title.set_text(current_plot_title)
            ax.tick_params(axis='both', labelsize=5)

            if plot_config.plot_name is not None:
                plot_name = plot_config.plot_name
            else:
                plot_name = plot_config.type

            save_plot(fig, plot_name, shap_config)
            fig.clf()


def spatial_plot_geotiff(shap_vals, plot_config, shap_config, **kwargs):
    return None


def scatter_plot(shap_vals, plot_config, shap_config, **kwargs):
    if 'feature_names' in kwargs:
        feature_names = shap_vals.feature_names
    else:
        feature_names = [str(x) for x in range(shap_vals.shape[1])]

    if plot_config.plot_features is None:
        log.error('No plot features provided, cannot plot')
        return None

    fig, ax = plt.subplots(figsize=(1.920, 1.080), dpi=100)
    multi_output_dim = shap_vals.shape[2] if len(shap_vals.shape) else 1
    for dim_idx in range(multi_output_dim):
        for feat in plot_config.plot_features:
            inter_feat = feat[1] if len(feat) == 2 else 'auto'
            plot_vals = shap_vals[:, :, dim_idx]
            shap.dependence_plot(feat[0], plot_vals.values, plot_vals.data, feature_names=feature_names,
                                 interaction_index=inter_feat, show=False, ax=ax)

            current_plot_title = plot_config.plot_title if plot_config.plot_title is not None else ''
            add_on_string = f'_feature_{feature_names[feat_idx]}_output_{dim_idx}'
            current_plot_title = current_plot_title + add_on_string
            ax.title.set_text(current_plot_title)
            ax.tick_params(axis='both', labelsize=5)

            if plot_config.plot_name is not None:
                plot_name = plot_config.plot_name
            else:
                plot_name = plot_config.type

            save_plot(fig, plot_name, shap_config)
            fig.clf()


plotting_func_map = {
    'summary': summary_plot,
    'bar': bar_plot,
    'decision': decision_plot,
    'shap_corr': shap_corr_plot,
    'waterfall': waterfall_plot
}

plotting_type_map = {
    'summary': aggregate_subplot,
    'bar': aggregate_subplot,
    'decision': aggregate_separate,
    'shap_corr': aggregate_separate,
    'spatial': spatial_plot,
    'scatter': scatter_plot,
    'waterfall': individual_subplot
}


def generate_plots(plot_config_list, shap_vals, shap_config, **kwargs):
    if kwargs['feature_names'] is None:
        log.warning('Feature names not provided, plots might be confusing')
    else:
        shap_vals.feature_names = kwargs['feature_names']

    current_plot_idx = 1
    for current_plot_config in plot_config_list:
        progress_message = f'Generating plot {current_plot_idx} of {len(plot_config_list)}'
        print(progress_message)
        plot_vals = shap_vals
        if current_plot_config.output_idx is not None:
            plot_vals = shap_vals[:, :, current_plot_config.output_idx]

        plotting_type_map[current_plot_config.type](plot_vals, current_plot_config, shap_config, **kwargs)
        current_plot_idx += 1


def generate_plots_poly_point(name_list, shap_vals_dict, shap_vals_point, shap_config, **kwargs):
    if kwargs['feature_names'] is None:
        log.warning('Feature names not provided, plots might be confusing')
    else:
        feature_names = ['\n'.join(wrap(feat, 30)) for feat in kwargs['feature_names']]
        shap_vals_point.feature_names = feature_names
        for key, val in shap_vals_dict.items():
            val.feature_names = feature_names

    for idx, name in enumerate(name_list):
        print(f'Generating plot {idx+1} of {len(name_list)}')
        current_point_poly_vals = shap_vals_dict[name]
        current_point_vals = shap_vals_point[idx, :, :]
        current_point_vals.data = current_point_vals.data[idx, :]
        point_poly_subplots(name, current_point_poly_vals, current_point_vals, shap_config, **kwargs)


def ax_tidy_point_poly(target_ax, plot_title, padding=None):
    current_plot_title = '\n'.join(wrap(plot_title, 30))
    target_ax.set_title(current_plot_title, fontsize=7)
    target_ax.tick_params(axis='both', labelsize=3)
    target_ax.set_yticklabels(axs[0, 0].get_yticklabels(), rotation=45)
    target_ax.xaxis.get_label().set_fontsize(7)
    if padding is not None:
        target_ax.tick_params(axis='y', pad=padding)


def point_poly_subplots(name, point_poly_vals, point_vals, shap_config, **kwargs):
    num_plots = point_poly_vals.shape[2] if len(point_poly_vals.shape) > 2 else 1
    output_names = kwargs['output_names'] if 'output_names' in kwargs else None

    plot_width = 66
    plot_height = 66
    for plot_idx in range(num_plots):
        fig, axs = plt.subplots(2, 3, figsize=(plot_width, plot_height), dpi=250)
        current_output_name = output_names[plot_idx] if output_names is not None else plot_idx
        current_points_vals = point_vals[:, plot_idx]
        current_point_poly_vals = point_poly_vals[:, :, plot_idx]

        # Single prediction waterfall
        plt.sca(axs[0, 0])
        shap.waterfall_plot(current_points_vals, show=False)
        current_plot_title = f'Single Prediction Waterfall'
        ax_tidy_point_poly(axs[0, 0], current_plot_title, 10)

        # Multi prediction summary
        plt.sca(axs[1, 0])
        shap.summary_plot(current_point_poly_vals.values, features=current_point_poly_vals.data,
                          feature_names=current_point_poly_vals.feature_names, show=False,
                          plot_size=(plot_width, plot_height), color_bar=False)
        current_plot_title = f'Multi-Prediction Summary'
        ax_tidy_point_poly(axs[1, 0], current_plot_title)

        # Single prediction bar
        plt.sca(axs[0, 1])
        shap.plots.bar(current_points_vals, show=False)
        current_plot_title = f'Single Prediction Bar'
        ax_tidy_point_poly(axs[0, 1], current_plot_title, 10)

        # Multi prediction bar
        plt.sca(axs[1, 1])
        shap.plots.bar(current_point_poly_vals, show=False)
        current_plot_title = f'Multi-Prediction Bar'
        axs[1, 1].xaxis.get_label().set_fontsize(7)
        ax_tidy_point_poly(axs[1, 1], current_plot_title, 10)

        # Single prediction decision
        plt.sca(axs[0, 2])
        shap.decision_plot(current_points_vals.base_values, current_points_vals.values,
                           feature_names=current_points_vals.feature_names, auto_size_plot=False)
        current_plot_title = f'Single Prediction Decision'
        ax_tidy_point_poly(axs[0, 2], current_plot_title)

        # Multi prediction decision
        plt.sca(axs[1, 2])
        shap.decision_plot(current_point_poly_vals.base_values[0], current_point_poly_vals.values,
                           feature_names=current_point_poly_vals.feature_names, auto_size_plot=False)
        current_plot_title = f'Multi-Prediction Decision'
        ax_tidy_point_poly(axs[1, 2], current_plot_title)

        fig.suptitle(f'Single vs Multiple Point {name} Output {current_output_name}')

        plot_name = f'poly_point_{name}_{current_output_name}'
        Path(shap_config.output_path).mkdir(parents=True, exist_ok=True)
        plot_save_path = path.join(shap_config.output_path, plot_name + '.png')
        fig.tight_layout()
        fig.savefig(plot_save_path, dpi=250)
        plt.clf()
