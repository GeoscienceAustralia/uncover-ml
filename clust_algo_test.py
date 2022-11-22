import logging
import joblib
import os

from uncoverml import cluster
from uncoverml import config
from uncoverml import geoio
from uncoverml import predict
from uncoverml import mpiops
from uncoverml import features

from uncoverml.transforms import StandardiseTransform

log = logging.getLogger(__name__)


def prep_data(current_config, subsample_frac):
    for f in current_config.feature_sets:
        if not f.transform_set.global_transforms:
            raise ValueError('Standardise transform must be used for kmeans')
        for t in f.transform_set.global_transforms:
            if not isinstance(t, StandardiseTransform):
                raise ValueError('Only standardise transform is '
                                 'allowed for kmeans')

    current_config.subsample_fraction = subsample_frac
    if current_config.subsample_fraction < 1:
        print("Memory contstraint: using {:2.2f}%"
                 " of pixels".format(current_config.subsample_fraction * 100))
    else:
        print("Using memory aggressively: dividing all data between nodes")

    current_config.algorithm = current_config.clustering_algorithm
    current_config.cubist = False

    image_chunk_sets = geoio.unsupervised_feature_sets(current_config)

    transform_sets = [k.transform_set for k in current_config.feature_sets]
    current_features, _ = features.transform_features(image_chunk_sets,
                                                      transform_sets,
                                                      current_config.final_transform,
                                                      current_config)

    current_features, _ = features.remove_missing(current_features)

    raw_save_path = os.path.join(current_config.output_dir, 'raw_features.data')
    joblib.dump(image_chunk_sets, raw_save_path)

    features_save_path = os.path.join(current_config.output_dir, 'training_data.data')
    joblib.dump(current_features, features_save_path)

    return current_features


def model_train(model_type, main_config, x_train):
    current_model = None
    if model_type == 'kmeans':
        current_model = cluster.KMeans(main_config.n_classes, main_config.oversample_factor)
    elif model_type == 'dbscan':
        current_model = cluster.DBScan()
    elif model_type == 'hdbscan':
        current_model = cluster.HDBScan()

    if current_model:
        current_model.learn(x_train)
    else:
        raise TypeError('Clustering model is NoneType, fix this')

    return current_model


def model_predict(model, algo_type_string, main_config, partitions=1, mask=None, retain=None):
    main_config.cluster = True
    main_config.mask = mask if mask else main_config.mask
    if main_config.mask:
        main_config.retain = retain if retain else main_config.retain

        if not os.path.isfile(main_config.mask):
            main_config.mask = ''
            print('A mask was provided, but the file does not exist on '
                     'disc or is not a file.')

    main_config.n_subchunks = partitions
    if main_config.n_subchunks > 1:
        print("Memory contstraint forcing {} iterations "
                 "through data".format(main_config.n_subchunks))
    else:
        print("Using memory aggressively: dividing all data between nodes")

    image_shape, image_bbox, image_crs = geoio.get_image_spec(model, main_config)

    outfile_tif = algo_type_string
    predict_tags = model.get_predict_tags()

    if hasattr(main_config, 'outbands'):
        if not main_config.outbands:
            main_config.outbands = len(predict_tags)
    else:
        main_config.outbands = len(predict_tags)

    if not hasattr(main_config, 'geotif_options'):
        main_config.geotif_options = {}

    if not hasattr(main_config, 'thumbnails'):
        main_config.thumbnails = 10

    image_out = geoio.ImageWriter(image_shape, image_bbox, image_crs,
                                  outfile_tif,
                                  main_config.n_subchunks, main_config.output_dir,
                                  band_tags=predict_tags[
                                            0: min(len(predict_tags),
                                                   main_config.outbands)],
                                  **main_config.geotif_options)

    for i in range(main_config.n_subchunks):
        print("starting to render partition {}".format(i + 1))
        predict.render_partition(model, i, image_out, main_config)

    # explicitly close output rasters
    image_out.close()

    if main_config.cluster and main_config.cluster_analysis:
        if mpiops.chunk_index == 0:
            predict.final_cluster_analysis(main_config.n_classes,
                                           main_config.n_subchunks)

    # ls.predict.final_cluster_analysis(config.n_classes,
    #                                   config.n_subchunks)

    if main_config.thumbnails:
        image_out.output_thumbnails(main_config.thumbnails)
    print("Finished! Total mem = {:.1f} GB".format(_total_gb()))


def train_predict_models(model_list, config_file, subsample_frac, partitions=1, mask=None, retain=None):
    main_config = config.Config(config_file)
    x_train = prep_data(main_config, subsample_frac)
    for current_model in model_list:
        print(f'train-predict started for {current_model["type"]}')
        main_config.output_dir = current_model['out_dir']
        trained_model = model_train(current_model['type'], main_config, x_train)
        model_predict(trained_model, current_model['type'], main_config, partitions, mask, retain)
        print(f'train-predict completed for {current_model["type"]}')


if __name__ == '__main__':
    mod_list = [
        {'type': 'hdbscan', 'out_dir': './results/hdbscan'},
        {'type': 'kmeans', 'out_dir': './results/kmeans'},
    ]
    current_config_file = './cluster-test.yaml'
    subsample_frac = 0.001
    parts = 200
    train_predict_models(mod_list, current_config_file, subsample_frac, parts)
