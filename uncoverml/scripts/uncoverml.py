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

log = logging.getLogger(__name__)

_memory_overhead = 4.0


def compute_n_subchunks(memory_threshold):
    if memory_threshold is not None:
        per_node_threshold = memory_threshold / float(ls.mpiops.chunks)
        overhead_threshold = per_node_threshold / float(_memory_overhead)
        n_subchunks = max(1, round(1.0 / overhead_threshold))
        log.info("Memory contstraint forcing {} iterations through data".format(
            n_subchunks))
    else:
        log.info("Using memory aggressively: loading all data into each node")
        n_subchunks = 1
    return n_subchunks


class MPIStreamHandler(logging.StreamHandler):
    """
    Only logs messages from Node 0
    """
    def emit(self, record):
        if ls.mpiops.chunk_index == 0:
            super().emit(record)


class ElapsedFormatter():

    def format(self, record):
        lvl = record.levelname
        name = record.name
        t = int(round(record.relativeCreated/1000.0))
        msg = record.getMessage()
        return "+{}s {}:{} {}".format(t, name, lvl, msg)


def configure_logging(verbosity):
    log = logging.getLogger("")
    log.setLevel(verbosity)
    ch = logging.StreamHandler()
    ch = MPIStreamHandler()
    formatter = ElapsedFormatter()
    ch.setFormatter(formatter)
    log.addHandler(ch)


@click.group()
@click.option('-v', '--verbosity', default='INFO', help='Level of logging')
def cli(verbosity):
    configure_logging(verbosity)


@cli.command()
@click.argument('pipeline_file')
@click.option('-m', '--memlimit', type=float, default=None,
              help='Try to less memory than this fraction of the input data')
def learn(pipeline_file, memlimit):
    config = ls.config.Config(pipeline_file)
    config.n_subchunks = compute_n_subchunks(memlimit)

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
def cluster(pipeline_file):
    config = ls.config.Config(pipeline_file)
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
@click.option('-m', '--memlimit', type=float, default=None,
              help='Try to less memory than this fraction of the input data')
def predict(model_or_cluster_file, memlimit):

    with open(model_or_cluster_file, 'rb') as f:
        state_dict = pickle.load(f)

    model = state_dict["model"]
    config = state_dict["config"]
    config.n_subchunks = compute_n_subchunks(memlimit)

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
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def hello(count, name):
    for x in range(count):
        click.echo('Hello %s!' % name)
