import sys
import pickle
import logging

import numpy as np
import click

import uncoverml as ls
import uncoverml.geoio
import uncoverml.features
import uncoverml.config
import uncoverml.learn
import uncoverml.cluster
import uncoverml.predict
import uncoverml.mpiops
import uncoverml.validate
import uncoverml.logging
import uncoverml.memory

log = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def cli(verbosity):
    ls.logging.configure(verbosity)


@cli.command()
@click.argument('pipeline_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
def learn(pipeline_file, partitions):
    config = ls.config.Config(pipeline_file)
    config.n_subchunks = partitions
    if config.n_subchunks > 1:
        log.info("Memory contstraint forcing {} iterations "
                 "through data".format(config.n_subchunks))
    else:
        log.info("Using memory aggressively: dividing all data between nodes")

    # Make the targets
    targets = ls.geoio.load_targets(shapefile=config.target_file,
                                    targetfield=config.target_property)
    # We're doing local models at the moment
    targets_all = ls.targets.gather_targets(targets)

    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.geoio.image_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    if config.rank_features:
        measures, features, scores = ls.validate.local_rank_features(
            image_chunk_sets,
            transform_sets,
            targets_all,
            config)
        ls.mpiops.run_once(ls.geoio.export_feature_ranks, measures, features,
                           scores, config)

    x = ls.features.transform_features(image_chunk_sets, transform_sets,
                                       config.final_transform)
    # learn the model
    # local models need all data
    x_all = ls.features.gather_features(x)

    if config.cross_validate:
        crossval_results = ls.validate.local_crossval(x_all,
                                                      targets_all, config)
        ls.mpiops.run_once(ls.geoio.export_crossval, crossval_results, config)

    model = ls.learn.local_learn_model(x, targets, config)
    ls.mpiops.run_once(ls.geoio.export_model, model, config)


@cli.command()
@click.argument('pipeline_file')
@click.option('-s', '--subsample_fraction', type=float, default=1.0,
              help='only use this fraction of the data for learning classes')
def cluster(pipeline_file, subsample_fraction):
    config = ls.config.Config(pipeline_file)
    config.subsample_fraction = subsample_fraction
    if config.subsample_fraction < 1:
        log.info("Memory contstraint: using {:2.2f}%"
                 " of pixels".format(config.subsample_fraction * 100))
    else:
        log.info("Using memory aggressively: dividing all data between nodes")

    if config.semi_supervised:
        semisupervised(config)
    else:
        unsupervised(config)


def semisupervised(config):

    # make sure we're clear that we're clustering
    config.algorithm = config.clustering_algorithm
    # Get the taregts
    targets = ls.geoio.load_targets(shapefile=config.class_file,
                                    targetfield=config.class_property)

    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.geoio.semisupervised_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    x = ls.features.transform_features(image_chunk_sets, transform_sets,
                                       config.final_transform)

    x, classes = ls.features.remove_missing(x, targets)
    indices = np.arange(classes.shape[0], dtype=int)

    k = ls.cluster.compute_n_classes(classes, config)
    model = ls.cluster.KMeans(k, config.oversample_factor)
    log.info("Clustering image")
    model.learn(x, indices, classes)
    ls.mpiops.run_once(ls.geoio.export_cluster_model, model, config)


def unsupervised(config):
    # make sure we're clear that we're clustering
    config.algorithm = config.clustering_algorithm
    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.geoio.unsupervised_feature_sets(config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    x = ls.features.transform_features(image_chunk_sets, transform_sets,
                                       config.final_transform)

    x, _ = ls.features.remove_missing(x)
    k = config.n_classes
    model = ls.cluster.KMeans(k, config.oversample_factor)
    log.info("Clustering image")
    model.learn(x)
    ls.mpiops.run_once(ls.geoio.export_cluster_model, model, config)


@cli.command()
@click.argument('model_or_cluster_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
def predict(model_or_cluster_file, partitions):

    with open(model_or_cluster_file, 'rb') as f:
        state_dict = pickle.load(f)

    model = state_dict["model"]
    config = state_dict["config"]
    config.n_subchunks = partitions
    if config.n_subchunks > 1:
        log.info("Memory contstraint forcing {} iterations "
                 "through data".format(config.n_subchunks))
    else:
        log.info("Using memory aggressively: dividing all data between nodes")

    image_shape, image_bbox = ls.geoio.get_image_spec(model, config)

    outfile_tif = config.name + "_" + config.algorithm
    image_out = ls.geoio.ImageWriter(image_shape, image_bbox, outfile_tif,
                                     config.n_subchunks, config.output_dir,
                                     band_tags=model.get_predict_tags())

    for i in range(config.n_subchunks):
        log.info("starting to render partition {}".format(i+1))
        ls.predict.render_partition(model, i, image_out, config)
    log.info("Finished!")


@cli.command()
@click.argument('pipeline_file')
@click.option('-o', '--overhead', type=int, default=2,
              help='Estimate of memory overhead as multiplier')
@click.option('-s', '--subsample_fraction', type=float, default=1.0,
              help='only use this fraction of the data for clustering')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
def memory(pipeline_file, overhead, subsample_fraction, partitions):
    if ls.mpiops.chunks > 1:
        log.error("Please run this utility without MPI")
        sys.exit()

    config = ls.config.Config(pipeline_file)
    results = ls.memory.estimate(config, partitions,
                                 subsample_fraction, overhead)

    def fm(x, y):
        return x + ": {:2.2f}GB".format(y)

    output_string = ("\nMaximum Memory Usage Estimates" +
                     " with {} Partitions and {} cluster subsampling".format(
                        partitions, subsample_fraction) + ":\n\n" +
                     fm("Learning", results['learning']) + "\n" +
                     fm("Prediction", results['prediction']) + "\n" +
                     fm("Clustering (extraction)",
                        results['clustering-extraction']) + "\n" +
                     fm("Clustering (iteration)",
                        results['clustering-iteration']) + "\n" +
                     "\nNOTE:\n- Use more partitions to " +
                     "decrease memory usage for learning and prediction.\n" +
                     "- Use a lower subsampling fraction to decrease memory" +
                     " usage for clustering (iteration).\n" +
                     "- Total memory usage for clustering (extraction) " +
                     "cannot be decreased.")
    print(output_string)
