import os
import csv
import rasterio
import math
import zipfile
import requests
import json

import numpy as np

from pathlib import Path
from PIL import Image


def rename_files_before_upload(config):
    job_results_dir = Path(config.output_dir)
    all_paths = job_results_dir.glob('**/*')
    all_files = [f for f in all_paths if f.is_file()]
    all_files_ex_model = [f for f in all_files if f.name != 'config.model']
    for current_file in all_files_ex_model:
        # hard coding this for now, will need to generalise later for other models - Adi
        if 'transformedrandomforest_' in current_file.name:
            new_file_name = current_file.name.replace('transformedrandomforest_', '')
            current_file.rename(job_results_dir / new_file_name)
            print(str(job_results_dir / new_file_name))


def calc_std(config):
    # This loads an array into memory and does the calculation
    # This approach should be suitable for small prediction areas
    # Can adapt this later to be calculated in chunks, if needed
    # - Adi
    res_path = Path(config.output_dir)
    in_file = res_path / 'variance.tif'
    with rasterio.open(in_file, 'r') as src:
        var_data = src.read(1)
        var_profile = src.profile

    std_data = np.sqrt(var_data)
    out_file = res_path / 'std.tif'
    with rasterio.open(out_file, 'w', **var_profile) as dst:
        dst.write(std_data, 1)


def calc_uncert(config):
    res_path = Path(config.output_dir)
    pred_file = res_path / 'prediction.tif'
    with rasterio.open(pred_file, 'r') as pred_src:
        pred_data = pred_src.read(1, masked=True)
        pred_profile = pred_src.profile

    res_path = Path(config.output_dir)
    var_file = res_path / 'variance.tif'
    with rasterio.open(var_file, 'r') as var_src:
        var_data = var_src.read(1, masked=True)

    pred_data = pred_data.filled(np.nan)
    var_data = var_data.filled(np.nan)
    out_data = (4*np.sqrt(var_data))/pred_data

    out_file = res_path / 'uncert.tif'
    with rasterio.open(out_file, 'w', **pred_profile) as dst:
        dst.write(out_data, 1)


def stretch_raster(data, pct_lims=[5, 95]):
    percentiles = np.nanpercentile(np.ravel(data), pct_lims)
    percentiles = percentiles.tolist()
    lower_pct = percentiles[0]
    upper_pct = percentiles[1]

    data = data.astype(float)
    data = (data - lower_pct) / (upper_pct - lower_pct)
    data[np.isnan(data)] = 0
    data = np.maximum(np.minimum(data * 255, 255), 0)
    return data


def create_thumbnail(config, res_type):
    res_dir = Path(config.output_dir)
    in_file = res_dir / f'{res_type}.tif'
    with rasterio.open(str(in_file), 'r+') as src:
        src.nodata = np.nan
        data = src.read(1)

    stretched_data = stretch_raster(data)
    out_image = Image.fromarray(stretched_data.astype('uint8'), 'L')
    out_file = res_dir / f'{res_type}_thumbnail.png'
    out_image.save(str(out_file))
    print(str(out_file))


def create_results_zip(config):
    res_dir = Path(config.output_dir)

    tmp_out_dir = res_dir.parent
    out_file = f'{tmp_out_dir.as_posix()}/full_results.zip'
    with zipfile.ZipFile(out_file, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for fi in res_dir.iterdir():
            zip_ref.write(fi.as_posix(), arcname=fi.name)

    zip_out = Path(out_file)
    zip_out.rename(res_dir / zip_out.name)
    print(zip_out.as_posix())


# Need to add a function to the interface code that calls all of this code
# The interface side code needs to also upload
def read_presigned_urls_and_upload(config, job_type):
    res_dir = Path(config.output_dir)
    parent_dir = res_dir.parent
    json_file_name = 'upload_urls_pred.json' if job_type == 'pred' else 'upload_urls.json'
    upload_urls_file = parent_dir / json_file_name
    with open(str(upload_urls_file)) as json_urls:
        upload_urls_info = json.load(json_urls)

    files_uploaded = 0
    for url_info in upload_urls_info:
        # Get the name of the file to upload for the AWS key
        current_file_key = url_info['fields']['key']
        key_parts = current_file_key.split('/')
        upload_file_name = key_parts[-1]

        local_upload_file = res_dir / upload_file_name
        with open(str(local_upload_file), 'rb') as file_to_upload:
            upload_files = {'file': (str(local_upload_file), file_to_upload)}
            upload_resp = requests.post(url_info['url'], data=url_info['fields'],
                                        files=upload_files)

            print(f'Upload Response: {upload_resp.status_code}')
            if upload_resp.status_code == 204:
                files_uploaded += 1

    print(f'{files_uploaded} Files Uploaded')


