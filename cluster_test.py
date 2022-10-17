import logging

from uncoverml import cluster
from uncoverml import features
from uncoverml import mpiops
from uncoverml import config

from uncoverml.transforms import StandardiseTransform


log = logging.getLogger(__name__)


def training_data_plot(model_file, training_data_file):
    cluster.training_data_boxplot(model_file, training_data_file)


def prediction_data_plot(main_config):
    current_config = config.Config(main_config)
    cluster.all_feat_boxplot(current_config)


def feat_data_split_save(main_config):
    current_config = config.Config(main_config)
    cluster.split_all_feat_data(current_config)


def all_plots(model_file, training_data_file):
    cluster.generate_plots(model_file, training_data_file)


if __name__ == '__main__':
    current_model_file = './test_cluster.cluster'
    current_train_data = './training_data.data'
    all_plots(current_model_file, current_train_data)
