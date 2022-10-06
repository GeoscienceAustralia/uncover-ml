import logging

from uncoverml import cluster
from uncoverml import features
from uncoverml import mpiops
from uncoverml import config

from uncoverml.transforms import StandardiseTransform


log = logging.getLogger(__name__)


def kmean_analysis(model_file, training_data_file):
    cluster.training_data_boxplot(model_file, training_data_file)


if __name__ == '__main__':
    model_file = './results/test_cluster.cluster'
    training_data = './results/training_data.data'
    kmean_analysis(model_file, training_data)

