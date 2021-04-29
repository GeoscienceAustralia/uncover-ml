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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers.experimental import preprocessing

normalizer = preprocessing.Normalization()


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
X_train, y_train = create_train_test_set(data, data_line1, data_line2, data_line3, data_line4, data_line6)

# test using line 5
X_test, y_test = create_train_test_set(data, data_line5)


X_all, y_all = create_train_test_set(data, data_line1, data_line2, data_line3, data_line4, data_line5, data_line6)


train_features = X_train.copy()
test_features = X_test.copy()
# validation_features = X_val.copy()

normalizer.adapt(np.array(train_features))

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(train_features[:10])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error',
    metrics=[])

# history = linear_model.fit(
#     train_features, y_train,
#     epochs=100,
#     # suppress logging
#     verbose=2,
#     # Calculate validation results on 20% of the training data
#     validation_split=0.2)


def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(102, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    return model


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Ceno Depth]')
    plt.legend()
    plt.grid(True)


test_results = {}

# plot_loss(history)

test_results['linear_model'] = linear_model.evaluate(test_features, y_test, verbose=1)


dnn_model = build_and_compile_model(normalizer)

history = dnn_model.fit(
    train_features, y_train,
    validation_split=0.2,
    verbose=2, epochs=500)

plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, y_test, verbose=1)
print(pd.DataFrame(test_results, index=['Mean absolute error [Ceno Depth]']).T)

# print(r2_score(y_test, linear_model.predict(test_features)))
print('r2 score dnn: ', r2_score(y_test, dnn_model.predict(test_features)))

import IPython; IPython.embed(); import sys; sys.exit()
