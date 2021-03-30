import logging
from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from skopt.space import Real, Integer
from skopt import BayesSearchCV

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
aem_xy_and_other_covs, aem_conductivities, aem_thickness = extract_required_aem_data(original_aem_data, all_lines)

if not Path('covariates_targets.data').exists():
    data = convert_to_xy(aem_xy_and_other_covs, aem_conductivities, aem_thickness, all_lines)
    log.info("saving data on disc for future use")
    pickle.dump(data, open('covariates_targets.data', 'wb'))
else:
    log.warning("Reusing data from disc!!!")
    data = pickle.load(open('covariates_targets.data', 'rb'))

train_data = create_interp_data(all_interp_data, included_lines=[1, 2, 3, 4])
test_data = create_interp_data(all_interp_data, included_lines=[6])
validation_data = create_interp_data(all_interp_data, included_lines=[5])

X_train, y_train = create_train_test_set(data, test_data, validation_data)
X_test, y_test = create_train_test_set(data, train_data, validation_data)
X_val, y_val = create_train_test_set(data, train_data, test_data)

include_z = False
if include_z:
    X_train = X_train.drop('Z_coor')
    X_test = X_test.drop('Z_coor')
    X_val = X_val.drop('Z_coor')


log.info(f"Train data size: {X_train.shape}, Test data size: {X_test.shape}")


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
    n_iter=60,
    cv=3,
    verbose=1000,
    n_points=12,
    n_jobs=4,
    scoring=my_custom_scorer
)

# callback handler
def on_step(optim_result):
    score = searchcv.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True

searchcv.fit(X_train, y_train, callback=on_step)
print(searchcv.score(X_val, y_val))
import time
pickle.dump(searchcv, open(f"{reg.__class__.__name__}.{int(time.time())}.model", 'wb'))
import IPython; IPython.embed(); import sys; sys.exit()