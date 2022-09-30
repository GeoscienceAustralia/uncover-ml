import logging

from uncoverml import cluster
from uncoverml import features
from uncoverml import mpiops
from uncoverml import config

from uncoverml.transforms import StandardiseTransform


log = logging.getLogger(__name__)


def kmeans(config_file, subsample_fraction):
    current_config = config.Config(config_file)

    for f in current_config.feature_sets:
        if not f.transform_set.global_transforms:
            raise ValueError('Standardise transform must be used for kmeans')
        for t in f.transform_set.global_transforms:
            if not isinstance(t, StandardiseTransform):
                raise ValueError('Only standardise transform is '
                                 'allowed for kmeans')

    current_config.subsample_fraction = subsample_fraction
    if current_config.subsample_fraction < 1:
        log.info("Memory contstraint: using {:2.2f}%"
                 " of pixels".format(current_config.subsample_fraction * 100))
    else:
        log.info("Using memory aggressively: dividing all data between nodes")

    image_chunk_sets = ls.geoio.unsupervised_feature_sets(config)
    transform_sets = [k.transform_set for k in current_config.feature_sets]
    current_features, _ = features.transform_features(image_chunk_sets,
                                                      transform_sets,
                                                      current_config.final_transform,
                                                      current_config)
    current_features, _ = features.remove_missing(current_features)
    model = cluster.KMeans(current_config.n_classes, current_config.oversample_factor)
    print("Clustering image")
    model.learn(features)
    print('Generating plots')
    cluster.generate_save_plots(current_config, features, model, lon_lat)
    print('Plotting done, saving model')
    mpiops.run_once(ls.geoio.export_cluster_model, model, current_config)


if __name__ == '__main__':
    cluster_config = './cluster_test.yaml'
    subsample_frac = 1
    kmeans(cluster_config, subsample_frac)
