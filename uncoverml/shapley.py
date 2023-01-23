import math
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
    # Intersect polygon or single point
    # Inputs
    #   single_row_df - single row from geopandas dataframe containing geometry to intersect
    #   image_source_dir - file path to feature geotiff
    #
    # Outputs
    #   out_image - numpy array with values for intersected pixels
    #   lon_lats - numpy array of coordinates for intersected pixels
    geoms = single_row_df.geometry.values[0]
    geoms = [mapping(geoms)]
    # extract the raster values within the polygon
    with rasterio.open(image_source_dir) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        no_data = src.nodata

    data = out_image[0]
    shape = data.shape
    lon_lat = None
    if kwargs['type'] == 'poly':
        row, col = np.where(~np.isnan(data[0]))
        def rc2xy(r, c): return rasterio.transform.xy(out_transform, r, c, offset='center')
        v_func = np.vectorize(rc2xy)
        lon_lat = v_func(row, col)

    return out_image, lon_lat, shape


def get_data_points(loaded_shapefile, image_source):
    '''
    Intersect multiple points and combine the result
    Inputs
        loaded_shapefile - geopandas df with intersection geometries
        image_source - path to feature geotiff

    Outputs
        res_list - numpy array of concatenated pixel values for points
    '''
    res_list = []
    for idx, row in loaded_shapefile.iterrows():
        single_row_df = loaded_shapefile.iloc[[idx]]
        (result, lon_lat, shape) = intersect_shp(single_row_df, image_source, type='points')
        res_list.append(result)

    return np.concatenate(res_list)


def get_data_polygon(loaded_shapefile, image_source):
    '''
    Intersect single polygon
    Inputs
        loaded_shapefile - geopandas df with intersection geometries
        image_source - path to feature geotiff

    Outputs
        result - numpy array of intersected pixel values for polygon
        lon_lat - numpy array coordinates for intersected pixels
        shape - shape of intersected numpy array in pixels
    '''
    (result, lon_lat, shape) = intersect_shp(loaded_shapefile, image_source, type='poly')
    return result, lon_lat, shape


def image_feature_sets_shap(shap_config, main_config):
    '''
    Perform geometry intersection for all features
    Inputs
        shap_config - shap calclation config
        main_config - main uncoverml model config

    Outputs
        results - order dictionary of intersected numpy arrays for each feature
        OPTIONAL name_list - list of point names for point calculation
        OPTIONAL coords - dictionary of coordinate arrays for each intersected feature
    '''
    loaded_shapefile = gpd.read_file(shap_config.shapefile['dir'])
    name_list = None
    if shap_config.shapefile['type'] == 'points':
        name_list = loaded_shapefile['Name'].to_list()

    results = []
    coords = {}
    for s in main_config.feature_sets:
        extracted_chunks = {}
        for tif in s.files:
            name = path.abspath(tif)
            if shap_config.shapefile['type'] == 'points':
                x = get_data_points(loaded_shapefile, name)
            else:
                x, lon_lats, shape = get_data_polygon(loaded_shapefile, name)
                coord_name = tif.replace(shap_config.feature_path, '').replace('.tif', '')
                coords[coord_name] = (lon_lats, shape)

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

    if shap_config.shapefile['type'] == 'points':
        return results, name_list
    else:
        return results, coords


def load_data_shap(shap_config, main_config):
    '''
        Intersect features, transform and return
        Inputs
            shap_config - shap calclation config
            main_config - main uncoverml model config

        Outputs
            x_all - numpy array of transformed feature values
            OPTIONAL name_list - list of point names for point calculation
            OPTIONAL coords - dictionary of coordinate arrays for each intersected feature
        '''
    image_chunk_sets = image_feature_sets_shap(shap_config, main_config)
    name_list = None
    coords = None
    if shap_config.shapefile['type'] == 'points':
        image_chunk_sets, name_list = image_chunk_sets
    else:
        image_chunk_sets, coords = image_chunk_sets

    transform_sets = [k.transform_set for k in main_config.feature_sets]
    transformed_features, keep = features.transform_features(image_chunk_sets,
                                                             transform_sets,
                                                             main_config.final_transform,
                                                             main_config)
    x_all = features.gather_features(transformed_features[keep], node=0)

    if shap_config.shapefile['type'] == 'points':
        return x_all, name_list
    else:
        return x_all, coords


def load_point_poly_data(shap_config, main_config):
    '''
    Loop through each named point and get the feature data for that point
    Inputs
        shap_config - shap calclation config
        main_config - main uncoverml model config

    Outputs
        out_result - numpy array for intersected and expanded pixel window
        out_coords - numpy array of coordinates for area in out_result
    '''
    loaded_shapefile = gpd.read_file(shap_config.shapefile['dir'])
    name_list = loaded_shapefile['Name'].to_list()

    out_result = {}
    out_coords = {}
    for name in name_list:
        log.info(f'Getting data for {name}')
        current_row = loaded_shapefile[loaded_shapefile['Name'] == name]
        current_poly_data, current_coords = gen_poly_data(current_row, shap_config, main_config)
        out_result[name] = current_poly_data
        out_coords[name] = current_coords

    return out_result, out_coords


def gen_poly_data(single_row_df, shap_config, main_config):
    '''
    Get data and transform for a single named point
    Inputs
        single_row_df - single row from geopandas dataframe representing single named point
        shap_config - shap calclation config
        main_config - main uncoverml model config

    Outputs
        x_all - numpy array of intersected, expanded, transformed feature data
        coords - numpy array of coords for area represented by x_all
    '''
    size = shap_config.shapefile['size']
    image_chunk_sets, coords = gen_poly_from_point(single_row_df, main_config, size, shap_config)
    transform_sets = [k.transform_set for k in main_config.feature_sets]
    transformed_features, keep = features.transform_features(image_chunk_sets,
                                                             transform_sets,
                                                             main_config.final_transform,
                                                             main_config)
    x_all = features.gather_features(transformed_features[keep], node=0)
    return x_all, coords


def gen_poly_from_point(single_row_df, main_config, size, shap_config):
    '''
    Loop through features, intersect point and create expanded area for a given
    named point
    Inputs
        single_row_df - single row from geopandas dataframe representing single named point
        main_config - main uncoverml model config
        size - size for creation of size x size pixel grid area
        shap_config - shap calclation config

    Outputs
        results - numpy array of intersected and expanded data
        coords - coordinates representing area in results
    '''
    results = []
    coords = {}
    for s in main_config.feature_sets:
        extracted_chunks = {}
        for tif in s.files:
            name = path.abspath(tif)
            x, lon_lats = intersect_point_neighbourhood(single_row_df, size, name)
            coord_name = tif.replace(shap_config.feature_path, '').replace('.tif', '')
            coords[coord_name] = lon_lats
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

    return results, coords


def intersect_point_neighbourhood(single_row_df, size, image_source_dir):
    '''
    Intersected point and expand grid around it
    Inputs
        single_row_df - single row from geopandas dataframe representing single named point
        size - size for creation of size x size pixel grid area
        image_source - path to feature geotiff

    Outputs
        out_image - numpy array with values for intersected pixels
        lon_lats - numpy array of coordinates for intersected pixels
    '''
    single_point = single_row_df.geometry.values[0]
    with rasterio.open(image_source_dir) as src:
        py, px = src.index(single_point.x, single_point.y)
        window = rasterio.windows.Window(px - size // 2, py - size // 2, size, size)
        out_image = src.read(window=window)
        win_transform = rasterio.windows.transform(window, src.transform)
        row, col = np.where(~np.isnan(out_image[0]))
        def rc2xy(r, c): return rasterio.transform.xy(win_transform, r, c, offset='center')
        v_func = np.vectorize(rc2xy)
        lon_lat = v_func(row, col)

    return out_image, lon_lat


class ShapConfig:
    # Class that contains needed information for shap calculation and plotting
    def __init__(self, yaml_file, main_config):
        # Load config YAML
        with open(yaml_file, 'r') as f:
            s = yaml.safe_load(f)
        self.name = path.basename(yaml_file).rsplit(".", 1)[0]

        # Get the explainer type
        if 'explainer' in s:
            self.explainer = s['explainer']
        else:
            self.explainer = None
            log.error('No explainer provided, cannot calculate Shapley values')

        # Get the shapefile representing calculation region
        if 'shapefile' in s:
            self.shapefile = s['shapefile']
        else:
            self.shapefile = None
            log.error('No shapefile provided, calculation will fail')

        # Additional explainer arguements
        self.explainer_kwargs = s['explainer_kwargs'] if 'explainer_kwargs' in s else None
        # Start and end row for calculation
        self.calc_start_row = s['calc_start_row'] if 'calc_start_row' in s else None
        self.calc_end_row = s['calc_end_row'] if 'calc_end_row' in s else None
        # Masker type
        self.masker = s['masker'] if 'masker' in s else None

        # Get the output path for saving results
        if 'output_path' in s:
            self.output_path = s['output_path']
        elif hasattr(main_config, 'output_dir'):
            self.output_path = path.join(main_config.output_dir, 'shap')
        else:
            self.output_path = None
            logging.error('No output path, exception will be thrown when saving')

        # Get list of plot config, TO BE REMOVED
        self.plot_config_list = None
        if 'plots' in s:
            plot_configs = s['plots']
            self.plot_config_list = [PlotConfig(p) for p in plot_configs]
        else:
            log.warning('No plots will be created')
            self.plot_config_list = None

        # Names of model prediction outputs
        self.output_names = s['output_names'] if 'output_names' in s else None
        # Path where feature files are saved
        self.feature_path = s['feature_path'] if 'feature_path' in s else None
        # File from which pre-calculated shap values can be loaded
        self.load_file = s['load_file'] if 'load_file' in s else None

        # Options for saving calculated shap values
        if 'save' in s:
            self.do_save = True
            self.save_name = s['save']['name']
        else:
            self.do_save = False

        # List of feature file names
        self.file_names = None
        self.set_file_names(main_config)

        # List of short feature names
        self.feature_names = None
        self.set_feature_names(s)

    def set_file_names(self, config):
        # Use the info from the main config to get feature file names
        # Inputs
        #   config - Main UncoverML config
        file_names = []
        for s in config.feature_sets:
            for tif in s.files:
                new_string = tif.replace(self.feature_path, '').replace('.tif', '')
                file_names.append(new_string)

        self.file_names = file_names

    def set_feature_names(self, yaml_file):
        # Add short feature names into shap config
        # Inputs
        #   yaml_file - YAML file containing shap config info
        if 'feature_names' in yaml_file:
            self.feature_names = yaml_file['feature_names']
        elif self.feature_path is not None:
            self.feature_names = self.file_names


# Dictionary mapping explainers by string
explainer_map = {
    'explainer': {'function': shap.Explainer,
                  'requirements': ['masker'],
                  'allowed': ['independent', 'partition', 'data', 'list']}
}

# Dictionary mapping maskers by string
masker_map = {
    'independent': shap.maskers.Independent,
    'partition': shap.maskers.Partition
}


def select_masker(mask_type, mask_data, mask_info=None):
    '''
    Initialise a masker from provided information
    Inputs
        mask_type - string of masker type
        mask_data - feature data to be used for masking
        mask_info - OPTIONAL additional information for creating masker

    Outputs
        masker - masker object to be used with explainer
    '''
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
    '''
    Extract and check masker info from shap config, use it to create a masker
    Inputs
        shap_config - shap config object
        x_data - numpy array of feature data

    Outputs
        mask_var - shapley masker object
    '''
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
    '''
    Gather required objects and variables to create required explainer
    Inputs
        shap_config - shap config object
        x_data - numpy array of feature data

    Outputs
        ret_val - 0 list of requested requirements
                  1 count of unfulfilled requirements
    '''
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
    '''
    Calculate shap vals
    Inputs
        model - UncoverML model object
        shap_config - shap config object
        x_data - feature data for calculation

    Outputs
        shap_vals - explanation object containing calculated shapley values
    '''
    def shap_predict(x):
        # Nested function to return predictions
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
        explainer_obj = explainer_map[shap_config.explainer]['function'](shap_predict, *reqs,
                                                                         **shap_config.explainer_kwargs)
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
    # Contains for plot info, to be removed
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
    # Plot saving function
    Path(shap_config.output_path).mkdir(parents=True, exist_ok=True)
    plot_save_path = path.join(shap_config.output_path, plot_name + '.png')
    plt.tight_layout(pad=3)
    fig.savefig(plot_save_path, dpi=100)


common_x_text_map = {
    'summary': 'SHAP value (impact on model output)',
    'bar': 'mean(|SHAP value|) (average impact on model output magnitude)'
}


def aggregate_subplot(plot_vals, plot_type, shap_config, **kwargs):
    num_plots = plot_vals.shape[2] if len(plot_vals.shape) > 2 else 1
    output_names = kwargs['output_names']
    if output_names is None:
        output_names = [str(i) for i in range(num_plots)]

    row_height = 0.4
    plot_height = (plot_vals.shape[1] * row_height) + 1.5
    plot_width = 20 if plot_type == 'bar' else 16
    fig, axs = plt.subplots(1, num_plots, dpi=100)
    for idx in range(num_plots):
        current_plot_data = plot_vals[:, :, idx] if num_plots > 1 else plot_vals
        agg_sub_map[plot_type](current_plot_data, axs[idx], idx, **kwargs, output_name=output_names[idx])

    fig.set_size_inches(plot_width, plot_height, forward=True)
    common_x_text = common_x_text_map[plot_type]
    fig.text(0.5, 0.01, common_x_text, ha='center')
    save_plot(fig, plot_type, shap_config)
    plt.clf()


def aggregate_separate(plot_vals, plot_type, shap_config, **kwargs):
    num_plots = plot_vals.shape[2] if len(plot_vals.shape) > 2 else 1
    output_names = kwargs['output_names']
    if output_names is None:
        output_names = [str(i) for i in range(num_plots)]

    fig, ax = plt.subplots(figsize=(1.920, 1.080), dpi=100, sharey=True)
    for idx in range(num_plots):
        current_plot_data = plot_vals[:, :, idx] if num_plots > 1 else plot_vals
        agg_sep_map[plot_type](current_plot_data, ax, idx, **kwargs, output_name=output_names[idx])

        save_plot(fig, plot_type, shap_config)
        plt.clf()


def summary_plot(plot_data, target_ax, plot_idx, **kwargs):
    plt.sca(target_ax)
    row_height = 0.4
    plot_height = (plot_data.shape[1] * row_height) + 1.5
    plot_width = 16
    shap.summary_plot(plot_data.values, features=plot_data.data, feature_names=plot_data.feature_names, show=False,
                      max_display=plot_data.shape[1], sort=False, plot_size=(plot_width, plot_height))

    x_axis = target_ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    if plot_idx > 0:
        target_ax.axes.yaxis.set_visible(False)

    plt.gcf().axes[-1].remove()
    output_name = kwargs['output_name']
    current_plot_title = f'Shap Correlation Output {output_name}'
    target_ax.title.set_text(current_plot_title)


def bar_plot(plot_data, target_ax, **kwargs):
    plt.sca(target_ax)
    shap.plots.bar(plot_data, show=False, max_display=plot_data.shape[1])
    # target_ax.tick_params(axis='both', labelsize=5)
    x_axis = target_ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)

    target_ax.tick_params(axis='both', labelsize=5)
    output_name = kwargs['output_name']
    current_plot_title = f'Shap Correlation Output {output_name}'
    target_ax.title.set_text(current_plot_title)


def shap_corr_plot(plot_data, target_ax, **kwargs):
    if 'feature_names' in kwargs:
        feature_names = plot_data.feature_names
    else:
        feature_names = [str(x) for x in range(plot_data.shape[1])]

    plot_dataframe = pd.DataFrame(plot_data.values, columns=feature_names)
    corr_matrix = plot_dataframe.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', fmt='.1g', annot=False, ax=target_ax)
    target_ax.tick_params(axis='both', labelsize=5)
    output_name = kwargs['output_name']
    current_plot_title = f'Shap Correlation Output {output_name}'
    target_ax.title.set_text(current_plot_title)


def decision_plot(plot_data, target_ax, **kwargs):
    shap.decision_plot(plot_data.base_values[0], plot_data.values, feature_names=plot_data.feature_names)
    target_ax.tick_params(axis='both', labelsize=5)
    output_name = kwargs['output_name']
    current_plot_title = f'Polygon Decision {output_name}'
    target_ax.title.set_text(current_plot_title)


def to_scientific_notation(number):
    a, b = '{:.4E}'.format(number).split('E')
    return '{:.5f}E{:+03d}'.format(float(a)/10, int(b)+1)


def spatial_plot(feature_name, target_ax, plot_vals, lon_lats, **kwargs):
    plot_arr = plot_vals.values
    if 'size' in kwargs:
        plot_arr = np.reshape(plot_arr, (kwargs['size'], kwargs['size']))
    im = target_ax.imshow(plot_arr, interpolation='nearest', cmap=plt.cm.get_cmap('jet'))

    max_lat = lon_lats[1].max()
    min_lat = lon_lats[1].min()
    lat_range = np.linspace(min_lat, max_lat, 2*plot_arr.shape[0])
    lat_range = list(lat_range)
    x_tick_labels = [to_scientific_notation(lat) for lat in lat_range]
    target_ax.set_xticklabels(x_tick_labels)

    min_lon = lon_lats[0].min()
    max_lon = lon_lats[0].max()
    lon_range = np.linspace(min_lon, max_lon, 2*plot_arr.shape[1])
    lon_range = list(lon_range)
    y_tick_labels = [to_scientific_notation(lon) for lon in lon_range]
    target_ax.set_yticklabels(y_tick_labels)

    target_ax.set_title(feature_name)

    return im


def aggregate_feature_subplots(shap_vals, plot_type, shap_config, lon_lats, **kwargs):
    num_fig = shap_vals.shape[2] if len(shap_vals.shape) > 2 else 1

    n_rows, n_cols = select_subplot_grid_dims(point_poly_vals.shape[1])
    plot_width = 5 * n_cols
    plot_height = 5 * n_rows
    for fig_idx in range(num_fig):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(plot_width, plot_height), dpi=250)
        if output_names is not None:
            current_output_name = output_names[fig_idx]
        else:
            current_output_name = fig_idx

        for feat_idx in range(shap_vals.shape[1]):
            current_feature_name = shap_vals.feature_names[feat_idx]
            current_plot_vals = shap_vals[:, feat_idx, fig_idx]
            current_lon_lats = lon_lats[shap_config.file_names[feat_idx]]

            if plot_type == 'spatial':
                dependence_plot(current_feature_name, current_plot_vals, np.ravel(axs)[feat_idx], **kwargs)
            elif plot_type == 'scatter':
                spatial_plot(current_feature_name, np.ravel(axs)[feat_idx], current_plot_vals, current_lon_lats,
                             **kwargs)

        fig.suptitle(f'Polygon {plot_type} plot Output {current_output_name}')
        plot_name = f'polygon_{plot_type}_{current_output_name}'
        Path(shap_config.output_path).mkdir(parents=True, exist_ok=True)
        plot_save_path = path.join(shap_config.output_path, plot_name + '.png')
        fig.tight_layout()
        fig.savefig(plot_save_path, dpi=250)
        plt.clf()


agg_sub_map = {
    'summary': summary_plot,
    'bar': bar_plot
}

agg_sep_map = {
    'decision': decision_plot,
    'shap_corr': shap_corr_plot
}


def generate_plots_poly(shap_vals, shap_config, lon_lats, **kwargs):
    if kwargs['feature_names'] is None:
        log.warning('Feature names not provided, plots might be confusing')
    else:
        shap_vals.feature_names = kwargs['feature_names']

    output_names = None
    if len(shap_vals.shape) > 2:
        if shap_config.output_names is not None:
            output_names = shap_config.output_names
        else:
            output_names = [str(i) for i in range(shap_vals.shape[2])]

    aggregate_subplot(shap_vals, 'summary', shap_config, **kwargs, output_names=output_names)
    aggregate_subplot(shap_vals, 'bar', shap_config, **kwargs, output_names=output_names)
    aggregate_separate(shap_vals, 'decision', shap_config, **kwargs, output_names=output_names)
    aggregate_separate(shap_vals, 'shap_corr', shap_config, **kwargs, output_names=output_names)
    aggregate_feature_subplots(shap_vals, 'scatter', shap_config, lon_lats, **kwargs, output_names=output_names)
    aggregate_feature_subplots(shap_vals, 'spatial', shap_config, lon_lats, **kwargs, output_names=output_names)


def generate_plots_poly_point(name_list, shap_vals_dict, shap_vals_point, shap_config, **kwargs):
    if shap_config.feature_names is None:
        log.warning('Feature names not provided, plots might be confusing')
    else:
        feature_names = ['\n'.join(wrap(feat, 30)) for feat in shap_config.feature_names]
        shap_vals_point.feature_names = feature_names
        for key, val in shap_vals_dict.items():
            val.feature_names = feature_names

    for idx, name in enumerate(name_list):
        print(f'Plotting point {idx + 1} of {len(name_list)}')
        current_point_poly_vals = shap_vals_dict[name]
        current_point_vals = shap_vals_point[idx]
        current_point_vals.data = current_point_vals.data[idx, :]
        point_poly_subplots(name, current_point_poly_vals, current_point_vals, shap_config, **kwargs)
        if 'lon_lats' in kwargs:
            log.info(f'Creating spatial plot for {name}')
            current_lon_lats = kwargs['lon_lats'][name]
            spatial_point_poly(name, current_point_poly_vals, current_lon_lats, shap_config,
                               output_names=kwargs['output_names'])


def ax_tidy_point_poly(target_ax, plot_title, padding=None):
    current_plot_title = '\n'.join(wrap(plot_title, 30))
    target_ax.set_title(current_plot_title, fontsize=7)
    target_ax.tick_params(axis='both', labelsize=5)
    target_ax.set_yticklabels(target_ax.get_yticklabels(), rotation=45)
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

        current_points_vals = point_vals
        current_point_poly_vals = point_poly_vals
        if num_plots > 1:
            current_points_vals = current_points_vals[:, plot_idx]
            current_point_poly_vals = current_point_poly_vals[:, :, plot_idx]

        if type(current_points_vals.base_values) == np.ndarray:
            if current_points_vals.base_values.size == 1:
                current_points_vals.base_values = current_points_vals.base_values[0]
            else:
                raise ValueError('Single base value is required for plotting')

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


def gen_factors(num):
    factors = []
    for i in range(1, int(num ** 0.5) + 1):
        if num % i == 0:
            factors.append((i, num / i))
    return factors


def select_subplot_grid_dims(num_subplots):
    grid_sizes = [n for n in range(num_subplots, num_subplots + 6)]
    n_rows = 1
    n_cols = num_subplots
    for size in grid_sizes:
        size_fac = gen_factors(size)
        diff = [abs(i[0] - i[1]) for i in size_fac]
        min_diff_loc = np.argmin(diff)
        if diff[min_diff_loc] < abs(n_rows - n_cols):
            n_rows = min(size_fac[min_diff_loc])
            n_cols = max(size_fac[min_diff_loc])

    return int(n_rows), int(n_cols)


def spatial_point_poly(name, point_poly_vals, lon_lats, shap_config, **kwargs):
    num_fig = point_poly_vals.shape[2] if len(point_poly_vals.shape) > 2 else 1
    output_names = kwargs['output_names'] if 'output_names' in kwargs else None

    n_rows, n_cols = select_subplot_grid_dims(point_poly_vals.shape[1])
    plot_width = 5 * n_cols
    plot_height = 5 * n_rows
    for fig_idx in range(num_fig):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(plot_width, plot_height), dpi=250)
        if output_names is not None:
            current_output_name = output_names[fig_idx]
        else:
            current_output_name = fig_idx

        for feat_idx in range(point_poly_vals.shape[1]):
            current_feature_name = point_poly_vals.feature_names[feat_idx]
            current_plot_vals = point_poly_vals[:, feat_idx, fig_idx]
            current_lon_lats = lon_lats[shap_config.file_names[feat_idx]]
            size = int(math.sqrt(current_plot_vals.shape[0]))
            im = spatial_plot(current_feature_name, np.ravel(axs)[feat_idx], current_plot_vals, current_lon_lats,
                             size=size)

        fig.colorbar(im, ax=axs.ravel().tolist())
        fig.suptitle(f'Multiple Point Spatial Plot {name} Output {current_output_name}')
        plot_name = f'spatial_poly_point_{name}_{current_output_name}'
        Path(shap_config.output_path).mkdir(parents=True, exist_ok=True)
        plot_save_path = path.join(shap_config.output_path, plot_name + '.png')
        fig.tight_layout()
        fig.savefig(plot_save_path, dpi=250)
        plt.clf()
