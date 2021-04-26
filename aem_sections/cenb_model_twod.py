import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from skopt.space import Real, Integer
from skopt import BayesSearchCV

from aem_sections import utils
from aem_sections.utils import extract_required_aem_data, convert_to_xy, create_interp_data, create_train_test_set

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

aem_folder = '/home/sudipta/Downloads/aem_sections'
log.info("reading interp data...")
all_interp_data = pd.read_csv(Path(aem_folder).joinpath('Albers_cenozoic_interp.txt'))

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(line1.X_coor, line1.Y_coor, line1.AusAEM_DEM - line1.DEPTH, 'gray')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('depth')
# plt.show()

# plt.plot(line1.d.values, line1.AusAEM_DEM - line1.DEPTH); plt.show()
# 1 there are two typesof CEN-B lines

# verification y's
# y_true = line.DEPTH

log.info("reading covariates")
original_aem_data = pd.read_csv(Path(aem_folder).joinpath('Albers_data_AEM_SB.csv'))

all_lines = create_interp_data(all_interp_data, included_lines=[1, 2, 3, 4, 5, 6])
aem_xy_and_other_covs, aem_conductivities, aem_thickness = extract_required_aem_data(
    original_aem_data, all_lines, twod=True, include_thickness=True, add_conductivity_derivative=True)

data_line1 = create_interp_data(all_interp_data, included_lines=[1])
data_line2 = create_interp_data(all_interp_data, included_lines=[2])
data_line3 = create_interp_data(all_interp_data, included_lines=[3])
data_line4 = create_interp_data(all_interp_data, included_lines=[4])
data_line5 = create_interp_data(all_interp_data, included_lines=[5])
data_line6 = create_interp_data(all_interp_data, included_lines=[6])

if not Path('covariates_targets_2d.data').exists():
    data = convert_to_xy(aem_xy_and_other_covs, aem_conductivities, aem_thickness, all_lines, twod=True)
    log.info("saving data on disc for future use")
    pickle.dump(data, open('covariates_targets_2d.data', 'wb'))
else:
    log.warning("Reusing data from disc!!!")
    data = pickle.load(open('covariates_targets_2d.data', 'rb'))

all_data_lines = [data_line1, data_line2, data_line3, data_line4, data_line5, data_line6]

# take out lines 4 and 5 from train data
X_train, y_train = create_train_test_set(data, data_line1, data_line2, data_line3, data_line4)

# test using line 5
X_test, y_test = create_train_test_set(data, data_line5)

# val using line 4
X_val, y_val = create_train_test_set(data, data_line4)

X_all, y_all = create_train_test_set(data, data_line1, data_line2, data_line3, data_line4, data_line5, data_line6)

log.info(f"Train data size: {X_train.shape}, "
         f"Test data size: {X_test.shape}, "
         f"Validation data size: {X_val.shape}")


log.info("assembled covariates and targets")

log.info("tuning model params ....")

n_features = X_train.shape[1]

gbm_space = {'max_depth': Integer(1, 15),
             'learning_rate': Real(10 ** -5, 10 ** 0, prior="log-uniform"),
             'max_features': Integer(1, n_features),
             'min_samples_split': Integer(2, 100),
             'min_samples_leaf': Integer(1, 100),
             'n_estimators': Integer(20, 200),
             }

xgb_space = {
    'max_depth': Integer(1, 15),
    'learning_rate': Real(10 ** -5, 10 ** 0, prior="log-uniform"),
    'n_estimators': Integer(20, 200),
    'min_child_weight': Integer(1, 10),
    'max_delta_step': Integer(0, 10),
    'gamma': Real(0, 0.5, prior="uniform"),
    'colsample_bytree': Real(0.3, 0.9, prior="uniform"),
    'subsample': Real(0.01, 1.0, prior='uniform'),
    'colsample_bylevel': Real(0.01, 1.0, prior='uniform'),
    'colsample_bynode': Real(0.01, 1.0, prior='uniform'),
    'reg_alpha': Real(1, 100, prior='uniform'),
    'reg_lambda': Real(0.01, 10, prior='log-uniform'),
}

# sklearn gbm
# reg = GradientBoostingRegressor(random_state=0)

# xgboost
reg = XGBRegressor(random_state=0)
# max_depth=3, learning_rate=1, n_estimators=100,
# verbosity=1, silent=None,
# objective="reg:linear", n_jobs=1, nthread=None, gamma=0,
# min_child_weight=1, max_delta_step=0, subsample=0.8, colsample_bytree=1,
# colsample_bylevel=1, colsample_bynode=0.8, reg_alpha=0, reg_lambda=1,
# scale_pos_weight=1, base_score=0.5, random_state=0, seed=None,
# missing=None


def my_custom_scorer(reg, X, y):
    """learn on train data and predict on test data to ensure total out of sample validation"""
    y_test_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)
    return r2


searchcv = BayesSearchCV(
    reg,
    search_spaces=gbm_space if isinstance(reg, GradientBoostingRegressor) else xgb_space,
    n_iter=180,
    cv=2,  # use 2 when using custom scoring using X_test
    verbose=1000,
    n_points=12,
    n_jobs=3,
    scoring=my_custom_scorer
)

# callback handler
def on_step(optim_result):
    score = searchcv.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True

# searchcv.fit(X_train, y_train, callback=on_step)
import time
# pickle.dump(searchcv, open(f"{reg.__class__.__name__}.{int(time.time())}.model", 'wb'))

searchcv = pickle.load(open("XGBRegressor.1617704729.model", 'rb'))  # validation  data sline 4
# searchcv = pickle.load(open("XGBRegressor.1618369530.model", 'rb'))  # validation line 6
print(searchcv.score(X_val, y_val))

from mpl_toolkits import mplot3d; import matplotlib.pyplot as plt
# plt.scatter(y_val, searchcv.predict(X_val))
# plt.xlabel('y_true')
# plt.ylabel('y_pred')


def plot_validation_line(X_val: pd.DataFrame, val_data_line: pd.DataFrame, model: BayesSearchCV):
    import matplotlib.pyplot as plt
    from aem_sections.utils import add_delta
    original_cols = X_train.columns[:]
    plt.figure()
    X_val = add_delta(X_val)
    origin = (X_val.X_coor.iat[0], X_val.Y_coor.iat[0])
    val_data_line = add_delta(val_data_line, origin=origin)
    plt.plot(X_val.d, model.predict(X_val[original_cols]), label='prediction')
    plt.plot(val_data_line.d, val_data_line.Z_coor, label='interpretation')
    plt.xlabel('distance')
    plt.ylabel('cen-b depth')
    plt.legend()

    plt.show()


def plot_2d_section(X: pd.DataFrame, val_data_line: pd.DataFrame, model: BayesSearchCV, col_name: str,
                    flip_column=False, v_min=0.3, v_max=0.8):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm, PowerNorm
    from matplotlib.colors import Colormap
    from aem_sections.utils import add_delta, conductivities, thickness
    original_cols = X.columns[:]
    X = add_delta(X)
    origin = (X.X_coor.iat[0], X.Y_coor.iat[0])
    val_data_line = add_delta(val_data_line, origin=origin)
    d_conduct_cols = ['d_' + c for c in conductivities]
    Z = X[conductivities]
    # Z = X[d_conduct_cols]
    # Z = Z - np.min(np.min((Z))) + 1.0e-5
    h = X[thickness]
    dd = X.d
    ddd = np.atleast_2d(dd).T
    d = np.repeat(ddd, h.shape[1], axis=1)
    fig, ax = plt.subplots(figsize=(40, 4))
    cmap = plt.get_cmap('viridis')

    # Normalize(vmin=0.3, vmax=0.6) d(cond) norm
    im = ax.pcolormesh(d, -h, Z, norm=LogNorm(), cmap=cmap, linewidth=1, rasterized=True)
    fig.colorbar(im, ax=ax)
    axs = ax.twinx()
    ax.plot(X.d, -model.predict(X[original_cols]), label='prediction', linewidth=2, color='r')
    ax.plot(val_data_line.d, -val_data_line.Z_coor, label='interpretation', linewidth=2, color='k')

    axs.plot(X.d, -X[col_name] if flip_column else X[col_name], label=col_name, linewidth=2, color='orange')

    ax.set_xlabel('distance along aem line (m)')
    ax.set_ylabel('depth (m)')
    plt.title("Conductivity vs depth")

    ax.legend()
    axs.legend()
    plt.show()


def interpolate_and_write_to_raster(d, h, Z):
    # interpolation  for raster
    from scipy import interpolate
    f = interpolate.CloughTocher2DInterpolator(np.array([d.flatten(), np.array(h).flatten()]).T, np.array(Z).flatten())
    d_new = np.arange(np.min(d), np.max(d), 1000)
    h_new = np.arange(np.min(np.array(h)), np.max(np.array(h)), 10)
    mesh = np.meshgrid(d_new, h_new)
    Z_int = f(*mesh)
    plt.contourf(*mesh, Z_int)
    import rasterio
    from rasterio.transform import Affine
    res_d = 1000
    res_h = 10
    transform = Affine.translation(h_new[0] - res_h / 2, d_new[0] - res_d / 2) * Affine.scale(res_d, res_h)
    with rasterio.open(
            f'new_{int(time.time())}.tif',
            'w',
            driver='GTiff',
            height=Z_int.shape[0],
            width=Z_int.shape[1],
            count=1,
            dtype=Z_int.dtype,
            crs='+proj=aea +lon_0=90w +lat_1=20n +lat_2=60n',
            transform=transform
    ) as dst:
        dst.write(Z_int, 1)


def plot_3d_validation_line(X_val):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(X_val.X_coor, X_val.Y_coor, X_val.DEPTH, 'gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('depth')
    plt.show()


def plot_feature_importance(X, y, optimised_model: BayesSearchCV):
    xgb_model = XGBRegressor(**optimised_model.best_params_)
    xgb_model.fit(X, y)
    non_zero_indices = xgb_model.feature_importances_ >= 0.001
    non_zero_cols = X_all.columns[non_zero_indices]
    non_zero_importances = xgb_model.feature_importances_[non_zero_indices]
    sorted_non_zero_indices = non_zero_importances.argsort()
    plt.barh(non_zero_cols[sorted_non_zero_indices], non_zero_importances[sorted_non_zero_indices])
    plt.xlabel("Xgboost Feature Importance")

import IPython; IPython.embed(); import sys; sys.exit()
