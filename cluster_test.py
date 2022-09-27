from uncoverml import cluster
from uncoverml import features
from uncoverml import mpiops


def kmeans(config_file):
    config = ls.config.Config(config_file)

    for f in config.feature_sets:
        if not f.transform_set.global_transforms:
            raise ValueError('Standardise transform must be used for kmeans')
        for t in f.transform_set.global_transforms:
            if not isinstance(t, StandardiseTransform):
                raise ValueError('Only standardise transform is '
                                 'allowed for kmeans')

    config.subsample_fraction = subsample_fraction
    if config.subsample_fraction < 1:
        log.info("Memory contstraint: using {:2.2f}%"
                 " of pixels".format(config.subsample_fraction * 100))
    else:
        log.info("Using memory aggressively: dividing all data between nodes")

    image_chunk_sets, lon_lat = cluster.extract_features_lon_lat(config)
    transform_sets = [k.transform_set for k in config.feature_sets]
    current_features, _ = features.transform_features(image_chunk_sets,
                                                      transform_sets,
                                                      config.final_transform,
                                                      config)
    current_features, _ = features.remove_missing(current_features)
    model = cluster.KMeans(config.n_classes, config.oversample_factor)
    print("Clustering image")
    model.learn(features)
    print('Generating plots')
    cluster.generate_save_plots(config, features, model, lon_lat)
    print('Plotting done, saving model')
    mpiops.run_once(ls.geoio.export_cluster_model, model, config)


