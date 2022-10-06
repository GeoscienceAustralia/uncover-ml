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


if __name__ == '__main__':
    config_file = './cluster_test.yaml'
    prediction_data_plot(config_file)

    # model_file = './results/test_cluster.cluster'
    # training_data = './results/training_data.data'
    # training_data_plot(model_file, training_data)

