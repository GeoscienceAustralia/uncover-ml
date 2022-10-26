import logging

from uncoverml import cluster
from uncoverml import features
from uncoverml import mpiops
from uncoverml import config

from uncoverml.transforms import StandardiseTransform


log = logging.getLogger(__name__)


def feat_data_split_save(main_config):
    current_config = config.Config(main_config)
    cluster.partial_split(current_config)


def all_plots(model_file, training_data_file):
    cluster.generate_plots(model_file, training_data_file)


if __name__ == '__main__':
    current_model_file = './results/test_cluster.cluster'
    current_train_data = './results/training_data.data'
    all_plots(current_model_file, current_train_data)

    # config_file = '/g/data/ge3/as6887/projects/uncoverml_models/cluster-test/cluster_test.yaml'
    # feat_data_split_save(config_file)
