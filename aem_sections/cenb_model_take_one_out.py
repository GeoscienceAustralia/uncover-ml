import logging
from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from skopt.space import Real, Integer
from skopt import BayesSearchCV

from aem_sections.utils import extract_required_aem_data, convert_to_xy, create_interp_data, aem_covariate_cols, \
    coords, threed_coords, covariate_cols_without_xyz, final_cols, extent_of_data, create_train_test_set

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

all_lines = create_interp_data(all_interp_data, [])
aem_xy_and_other_covs, aem_conductivities, aem_thickness = extract_required_aem_data(original_aem_data, all_lines)

if not Path('covariates_targets.data').exists():
    data = convert_to_xy(aem_xy_and_other_covs, aem_conductivities, aem_thickness, all_lines)
    log.info("saving data on disc for future use")
    pickle.dump(data, open('covariates_targets.data', 'wb'))
else:
    log.warning("Reusing data from disc!!!")
    data = pickle.load(open('covariates_targets.data', 'rb'))

all_X = data['covariates']
all_y = data['targets']


exclude_interp_data = create_interp_data(all_interp_data, holdout_line_nos=[1, 2, 3, 5, 6])
X_train, X_test, y_train, y_test = create_train_test_set(data, exclude_interp_data)
log.info(f"Train data size: {X_train.shape}, Test data size: {X_test.shape}")


log.info("assembled covariates and targets")

log.info("tuning model params ....")

n_features = X_train.shape[1]

space = {'max_depth': Integer(1, 15),
         'learning_rate': Real(10 ** -5, 10 ** 0, prior="log-uniform"),
         'max_features': Integer(1, n_features),
         'min_samples_split': Integer(2, 100),
         'min_samples_leaf': Integer(1, 100),
         'n_estimators': Integer(20, 200),
}

# sklearn gbm
reg = GradientBoostingRegressor(random_state=0)

searchcv = BayesSearchCV(
    reg,
    search_spaces=space,
    n_iter=60,
    cv=3,
    verbose=1000,
    n_points=12,
    n_jobs=4,
)


# callback handler
def on_step(optim_result):
    score = searchcv.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True

searchcv.fit(X_train, y_train, callback=on_step)
searchcv.score(X_test, y_test)
import IPython; IPython.embed(); import sys; sys.exit()