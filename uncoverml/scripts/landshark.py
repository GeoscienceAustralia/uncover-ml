import click
import logging

# landshark
import uncoverml as ls
import uncoverml.config
import uncoverml.geoio
import uncoverml.features
import uncoverml.learn


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument('pipeline_file')
def learn(pipeline_file):
    config = ls.config.Config(pipeline_file)

    # Make the targets
    targets = ls.geoio.load_targets(shapefile=config.target_file,
                                    targetfield=config.target_property)
    # We're doing local models at the moment
    targets_all = ls.targets.gather_targets(targets)

    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.features.image_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    if config.rank_features:
        measures, features, scores = ls.learn.local_rank_features(
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
def cluster():
    click.echo('Dropped the database')


def semisupervised(config):

    # make sure we're clear that we're clustering
    config.algorithm = config.clustering_algorithm
    # Get the taregts
    targets = geoio.load_targets(shapefile=config.class_file,
                                 targetfield=config.class_property)

    # Get the image chunks and their associated transforms
    image_chunk_sets = semisupervised_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    x = pipeline.transform_features(image_chunk_sets, transform_sets,
                                    config.final_transform)

    x, classes = remove_missing(x, targets)
    indices = np.arange(classes.shape[0], dtype=int)

    k = compute_n_classes(classes, config)
    model = cluster.KMeans(k, config.oversample_factor)
    log.info("Clustering image")
    model.learn(x, indices, classes)
    mpiops.run_once(export_cluster_model, model, config)


def unsupervised(config):
    # make sure we're clear that we're clustering
    config.algorithm = config.clustering_algorithm
    # Get the image chunks and their associated transforms
    image_chunk_sets = unsupervised_feature_sets(config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    x = pipeline.transform_features(image_chunk_sets, transform_sets,
                                    config.final_transform)

    x, _ = remove_missing(x)
    k = config.n_classes
    model = cluster.KMeans(k, config.oversample_factor)
    log.info("Clustering image")
    model.learn(x)
    mpiops.run_once(export_cluster_model, model, config)


@cli.command()
def predict(state_filename):
    
    with open(state_filename, 'rb') as f:
        state_dict = pickle.load(f)

    model = state_dict["model"]
    config = state_dict["config"]


    image_shape, image_bbox = get_image_spec(model, config)

    outfile_tif = config.name + "_" + config.algorithm
    image_out = geoio.ImageWriter(image_shape, image_bbox, outfile_tif,
                                  config.n_subchunks, config.output_dir,
                                  band_tags=model.get_predict_tags())

    for i in range(config.n_subchunks):
        log.info("starting to render partition {}".format(i+1))
        predict.render_partition(model, i, image_out, config)
    log.info("Finished!")


@cli.command()
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def hello(count, name):
    for x in range(count):
        click.echo('Hello %s!' % name)
