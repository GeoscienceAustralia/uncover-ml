import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

log = logging.getLogger(__name__)

aem_folder = '/home/sudipta/Downloads/aem_sections'
original_aem_data = pd.read_csv(Path(aem_folder).joinpath('Albers_data_AEM_SB.csv'))

# columns
conductivities = [c for c in original_aem_data.columns if c.startswith('conduct')]
thickness = [t for t in original_aem_data.columns if t.startswith('thick')]

# distance within which an interpretation point is considered to contribute to target values
radius = 200
cell_size = 10
dis_tol = 100  # meters, distance tolerance used
coords = ['X_coor', 'Y_coor']
threed_coords = coords + ['Z_coor']
aem_covariate_cols = ['ceno_euc_a', 'dem_fill', 'Gravity_la', 'national_W', 'relief_ele', 'relief_mrv', 'SagaWET9ce'] \
                     + ['elevation', 'tx_height']
# categorical = 'relief_mrv'
# covariate_cols_without_xyz = aem_covariate_cols + ['conductivity']
# final_cols = coords + aem_covariate_cols + ['Z_coor']


def extract_required_aem_data(in_scope_aem_data, interp_data):
    # find bounding box
    x_max, x_min, y_max, y_min = extent_of_data(interp_data)
    # use bbox to select data only for one line
    aem_data = in_scope_aem_data[
        (in_scope_aem_data.X_coor < x_max + dis_tol) & (in_scope_aem_data.X_coor > x_min - dis_tol) &
        (in_scope_aem_data.Y_coor < y_max + dis_tol) & (in_scope_aem_data.Y_coor > y_min - dis_tol)
        ]
    aem_data = aem_data.sort_values(by='Y_coor', ascending=False)
    aem_xy_and_other_covs = aem_data[coords + aem_covariate_cols]
    aem_conductivities = aem_data[conductivities]
    aem_thickness = aem_data[thickness].cumsum(axis=1)

    return aem_xy_and_other_covs, aem_conductivities, aem_thickness


def create_train_test_set(data, * excluded_interp_data):
    X = data['covariates']
    y = data['targets']
    excluded_indices = np.zeros(X.shape[0], dtype=bool)    # nothing is excluded
    for ex_data in excluded_interp_data:
        x_max, x_min, y_max, y_min = extent_of_data(ex_data)
        excluded_indices = excluded_indices | \
                           ((X.X_coor < x_max + dis_tol) & (X.X_coor > x_min - dis_tol) &
                            (X.Y_coor < y_max + dis_tol) & (X.Y_coor > y_min - dis_tol))

    return X[~excluded_indices], y[~excluded_indices]


def extent_of_data(data: pd.DataFrame) -> Tuple[float, float, float, float]:
    x_min, x_max = min(data.X_coor.values), max(data.X_coor.values)
    y_min, y_max = min(data.Y_coor.values), max(data.Y_coor.values)
    return x_max, x_min, y_max, y_min


def weighted_target(line_required: pd.DataFrame, tree: KDTree, x: np.ndarray):
    ind, dist = tree.query_radius(x, r=radius, return_distance=True)
    ind, dist = ind[0], dist[0]
    dist += 1e-6  # add just in case of we have a zero distance
    if len(dist):
        df = line_required.iloc[ind]
        weighted_depth = np.sum(df.Z_coor * (1 / dist) ** 2) / np.sum((1 / dist) ** 2)
        return weighted_depth
    else:
        return None


def convert_to_xy(aem_xy_and_other_covs, aem_conductivities, aem_thickness, interp_data):
    log.info("convert to xy and target values...")
    covariates_including_xyz = []
    tree = KDTree(interp_data)
    target_depths = []
    for xy, c, t in zip(aem_xy_and_other_covs.iterrows(), aem_conductivities.iterrows(), aem_thickness.iterrows()):
        i, covariates_including_xy_ = xy
        j, cc = c
        k, tt = t
        assert i == j == k
        for ccc, ttt in zip(cc, tt):
            covariates_including_xyz_ = covariates_including_xy_.append(
                pd.Series([ttt, ccc], index=['Z_coor', 'conductivity'])
            )
            x_y_z = covariates_including_xyz_[threed_coords].values.reshape(1, -1)

            y = weighted_target(interp_data, tree, x_y_z)
            if y is not None:
                covariates_including_xyz.append(covariates_including_xyz_)
                target_depths.append(y)
    X = pd.DataFrame(covariates_including_xyz)
    y = pd.Series(target_depths, name='target')
    return {'covariates': X, 'targets': y}


def create_interp_data(input_interp_data, included_lines):
    line = input_interp_data[(input_interp_data['Type'] != 'WITHIN_Cenozoic')
                             & (input_interp_data['line'].isin(included_lines))]
    line = add_delta(line)
    line = line.rename(columns={'DEPTH': 'Z_coor'})
    line_required = line[threed_coords]
    return line_required


def add_delta(line):
    line = line.sort_values(by='Y_coor', ascending=False)
    line['X_coor_diff'] = line['X_coor'].diff()
    line['Y_coor_diff'] = line['Y_coor'].diff()
    line['delta'] = np.sqrt(line.X_coor_diff ** 2 + line.Y_coor_diff ** 2)
    line['delta'] = line['delta'].fillna(value=0.0)
    line['d'] = line['delta'].cumsum()
    line = line.sort_values(by=['d'], ascending=True)
    return line
