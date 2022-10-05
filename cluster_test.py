import logging

from uncoverml import cluster
from uncoverml import features
from uncoverml import mpiops
from uncoverml import config

from uncoverml.transforms import StandardiseTransform


log = logging.getLogger(__name__)


def kmean_analysis(config_file):
    current_config = config.Config(config_file)
    cluster.generate_save_plots(current_config)
    print('Process complete')


if __name__ == '__main__':
    cluster_config = './cluster_test.yaml'
    kmean_analysis(cluster_config)

