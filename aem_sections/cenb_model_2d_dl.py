import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2_score_sklearn, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from skopt.space import Real, Integer, Categorical
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


from aem_sections.utils import extract_required_aem_data, convert_to_xy, create_interp_data, create_train_test_set

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

aem_folder = '/home/sudipta/Documents/new_datasets'
log.info("reading interp data...")
all_interp_data = gpd.GeoDataFrame.from_file(
    Path(aem_folder).joinpath('interpretation_zone53_albers_study_area_Ceno_depth.shp').as_posix()
)

log.info("reading covariates ...")
original_aem_data = gpd.GeoDataFrame.from_file(Path(aem_folder).joinpath('high_res_cond_clip_albers_skip_6.shp').as_posix())


# columns
conductivities = [c for c in original_aem_data.columns if c.startswith('cond')]
thickness = [t for t in original_aem_data.columns if t.startswith('thick')]

print(conductivities)
print(thickness)

line_col = 'SURVEY_LIN'
lines_in_data = np.unique(all_interp_data[line_col])
train_lines_in_data, test_lines_in_data = train_test_split(lines_in_data)

all_lines = create_interp_data(all_interp_data, included_lines=list(lines_in_data), line_col=line_col)
aem_xy_and_other_covs, aem_conductivities, aem_thickness = extract_required_aem_data(
    original_aem_data, all_lines, thickness, conductivities, twod=True, include_thickness=True,
    add_conductivity_derivative=True)


if not Path('covariates_targets_2d.data').exists():
    data = convert_to_xy(aem_xy_and_other_covs, aem_conductivities, aem_thickness, all_lines, twod=True)
    log.info("saving data on disc for future use")
    pickle.dump(data, open('covariates_targets_2d.data', 'wb'))
else:
    log.warning("Reusing data from disc!!!")
    data = pickle.load(open('covariates_targets_2d.data', 'rb'))

# import IPython; IPython.embed(); import sys; sys.exit()

train_data_lines = [create_interp_data(all_interp_data, included_lines=i, line_col=line_col) for i in train_lines_in_data]
test_data_lines = [create_interp_data(all_interp_data, included_lines=i, line_col=line_col) for i in test_lines_in_data]

all_data_lines = train_data_lines + test_data_lines

# take out lines 4 and 5 from train data
X_train, y_train = create_train_test_set(data, * train_data_lines)

# test using line 5
X_test, y_test = create_train_test_set(data, * test_data_lines)


X_all, y_all = create_train_test_set(data, * all_data_lines)


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

from tensorflow.keras.callbacks import LearningRateScheduler, History, EarlyStopping
np.random.seed(5)
epochs = 100
learning_rate = 0.1  # initial learning rate
decay_rate = 0.1
momentum = 0.8


def exp_decay(epoch):
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate

# learning schedule callback
loss_history = History()
lr_rate = LearningRateScheduler(exp_decay)
early_stopping = EarlyStopping(min_delta=1.0e-6, verbose=1, patience=3)
callbacks_list = [loss_history, lr_rate, early_stopping]

from tensorflow.keras import backend as K


def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


def build_and_compile_model(norm, activation, regularizer, input_dim, hidden_dim):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(input_dim, activation=activation, kernel_regularizer=regularizers.l2(regularizer)),
        layers.Dropout(0.5),
        layers.Dense(hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(regularizer)),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['mean_absolute_error', 'mean_squared_error', r2_score]
                  )
    return model


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Ceno Depth]')
    plt.legend()
    plt.grid(True)


def plot_r2_score_and_loss(history):
    # Plot the loss function
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(np.sqrt(history.history['loss']), 'r', label='train')
    ax.plot(np.sqrt(history.history['val_loss']), 'b', label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Loss', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)

    # Plot the r2 score
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(np.sqrt(history.history['r2_score']), 'r', label='train')
    ax.plot(np.sqrt(history.history['val_r2_score']), 'b', label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Accuracy', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)

    plt.grid(True)


test_results = {}

# plot_loss(history)

test_results['linear_model'] = linear_model.evaluate(test_features, y_test, verbose=1)


# dnn_model = build_and_compile_model(normalizer)
batch_size_factor = 10

# history = dnn_model.fit(
#     train_features, y_train,
#     validation_split=0.2,
#     batch_size=train_features.shape[0]//batch_size_factor,
#     callbacks=callbacks_list,
#     verbose=2, epochs=epochs)
#
# plot_loss(history)
#
# test_results['dnn_model'] = dnn_model.evaluate(test_features, y_test, verbose=1)
# # print(pd.DataFrame(test_results, index=['Mean absolute error [Ceno Depth]']).T)
#
# # print(r2_score(y_test, linear_model.predict(test_features)))
# print('r2 score dnn: ', r2_score_sklearn(y_test, dnn_model.predict(test_features)))
#
# import time
# # pickle.dump(searchcv, open(f"{reg.__class__.__name__}.{int(time.time())}.model", 'wb'))
# str_with_time = f"dnn.{int(time.time())}.model"
# Path('saved_model').mkdir(exist_ok=True)
# model_file_name = Path('saved_model').joinpath(str_with_time)
# dnn_model.save(model_file_name)
# import IPython; IPython.embed(); import sys; sys.exit()


# optimisation
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

model_CV = KerasRegressor(build_fn=build_and_compile_model, verbose=1)

dnn_space = {
    'activation': Categorical(categories=[]),
    'regularizer': Real(10 ** -5, 10 ** 0, prior="log-uniform"),
    'n_estimators': Integer(20, 200),

}


# activation, regularizer, n1, n2

# model_init_batch_epoch_CV = KerasClassifier(build_fn=create_model_2, verbose=1)

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
