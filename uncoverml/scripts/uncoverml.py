"""
Run the uncoverml pipeline for clustering, supervised learning and prediction.

.. program-output:: uncoverml --help
"""

import logging
import os
import sys

import joblib
import resource
import json
from os.path import isfile, splitext, exists
from pathlib import Path
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=Warning)

import click
import numpy as np
import matplotlib
matplotlib.use('Agg')

import uncoverml as ls
import uncoverml.cluster
import uncoverml.config
import uncoverml.features
import uncoverml.geoio
import uncoverml.learn
import uncoverml.mllog
import uncoverml.mpiops
import uncoverml.predict
import uncoverml.validate
import uncoverml.targets
import uncoveml.interface_utils
from uncoverml.transforms.linear import WhitenTransform
from uncoverml.transforms import StandardiseTransform
from uncoverml import optimisation, hyopt
# from uncoverml.mllog import warn_with_traceback
from uncoverml.scripts import superlearn_cli
from uncoverml.log_progress import write_progress_to_file


log = logging.getLogger(__name__)
# warnings.showwarning = warn_with_traceback


@click.group()
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def cli(verbosity):
    ls.mllog.configure(verbosity)


def run_crossval(x_all, targets_all, config):
    crossval_results = ls.validate.local_crossval(x_all, targets_all, config)
    ls.mpiops.run_once(ls.geoio.export_crossval, crossval_results, config)



@cli.command()
@click.argument('pipeline_file')
@click.option('-j', '--param_json', type=click.Path(exists=True), multiple=True,
              help='algorithm parameters json, possibly from a previous optimise job')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
def learn(pipeline_file, param_json, partitions):
    config = ls.config.Config(pipeline_file)
    if len(param_json) > 0:
        # log.info('the following jsons are used as params \n', click.echo('\n'.join(param_json)))
        param_dicts = []  # list of dicts
        for pj in param_json:
            with open(pj, 'r') as f:
                log.info(f"{config.algorithm} params were updated using {param_json}")
                param_dicts.append(json.load(f))
        config.algorithm_args = {k: v for D in param_dicts for k, v in D.items()}
    param_str = f'Learning {config.algorithm} model with the following params:\n'
    for param, value in config.algorithm_args.items():
        param_str += "{}\t= {}\n".format(param, value)

    log.info(f"{param_str}")

    write_progress_to_file('train', 'Loading targets', config)
    targets_all, x_all = _load_data(config, partitions)
    write_progress_to_file('train', 'Targets loaded', config)

    if config.cross_validate:
        write_progress_to_file('train', 'Running cross validation', config)
        run_crossval(x_all, targets_all, config)

    # Yes I know, log messages have been doubled up, will clean this up once
    # Everything works
    log.info("Learning full {} model".format(config.algorithm))
    write_progress_to_file('train', 'Learning full model', config)

    progress_file = Path(config.output_dir) / 'train_progress.txt'
    sys.stdout = open(str(progress_file), 'a')
    model = ls.learn.local_learn_model(x_all, targets_all, config)
    sys.stdout.close()
    write_progress_to_file('train', 'Model learning complete', config)

    write_progress_to_file('train', 'Exporting model', config)
    ls.mpiops.run_once(ls.geoio.export_model, model, config)
    write_progress_to_file('train', 'Model exported', config)

    # use trained model
    if config.permutation_importance:
        ls.mpiops.run_once(ls.validate.permutation_importance, model, x_all,
                           targets_all, config)

    # if config.feature_importance:
    #     # model, x_all, targets_all, config: Config
    #     ls.mpiops.run_once(ls.validate.plot_feature_importance, model, x_all,
    #                        targets_all, config)

    write_progress_to_file('train', 'Process complete, validating...', config)
    log.info("Finished! Total mem = {:.1f} GB".format(_total_gb()))


def _load_data(config: uncoverml.config.Config, partitions):
    if config.pickle_load:
        x_all = joblib.load(open(config.pickled_covariates, 'rb'))
        targets_all = joblib.load(open(config.pickled_targets, 'rb'))
        if config.cubist or config.multicubist:
            config.algorithm_args['feature_type'] = \
                joblib.load(open(config.featurevec, 'rb'))
        log.warning('Using  pickled targets and covariates. Make sure you have'
                    ' not changed targets file and/or covariates.')
    else:
        log.info('One or both pickled files were not '
                 'found. All targets will be intersected.')
        config.n_subchunks = partitions
        if config.n_subchunks > 1:
            log.info("Memory constraint forcing {} iterations "
                     "through data".format(config.n_subchunks))
        else:
            log.info("Using memory aggressively: "
                     "dividing all data between nodes")

        # Make the targets
        if config.train_data_pk and exists(config.train_data_pk):
            log.info('Reusing pickled training data')
            image_chunk_sets, transform_sets, targets = \
                joblib.load(open(config.train_data_pk, 'rb'))
        else:
            log.info('Intersecting targets as pickled train data was not '
                     'available')
            targets = ls.geoio.load_targets(shapefile=config.target_file,
                                            targetfield=config.target_property,
                                            conf=config)
            # Get the image chunks and their associated transforms
            image_chunk_sets = ls.geoio.image_feature_sets(targets, config)
            transform_sets = [k.transform_set for k in config.feature_sets]

        if config.rawcovariates:
            log.info('Saving raw data before any processing')
            ls.features.save_intersected_features_and_targets(image_chunk_sets,
                                                              transform_sets,
                                                              targets, config)

        if config.train_data_pk:
            joblib.dump([image_chunk_sets, transform_sets, targets],
                        open(config.train_data_pk, 'wb'))

        if config.rank_features:
            measures, features, scores = ls.validate.local_rank_features(
                image_chunk_sets,
                transform_sets,
                targets,
                config)
            ls.mpiops.run_once(ls.geoio.export_feature_ranks, measures,
                               features, scores, config)

        # need to add cubist cols to config.algorithm_args
        # keep: bool array corresponding to rows that are retained
        features, keep = ls.features.transform_features(image_chunk_sets,
                                                        transform_sets,
                                                        config.final_transform,
                                                        config)
        # learn the model
        # local models need all data
        x_all = ls.features.gather_features(features[keep], node=0)

        # We're doing local models at the moment
        targets_all = ls.targets.gather_targets(targets, keep, config, node=0)

        if config.pickle and ls.mpiops.chunk_index == 0:
            if hasattr(config, 'pickled_covariates'):
                joblib.dump(x_all, open(config.pickled_covariates, 'wb'))
            if hasattr(config, 'pickled_targets'):
                joblib.dump(targets_all, open(config.pickled_targets, 'wb'))

    return targets_all, x_all


@cli.command()
@click.argument('pipeline_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
def superlearn(pipeline_file: str, partitions: int):
    superlearn_cli.main(pipeline_file, partitions)


@cli.command()
@click.argument('pipeline_file', type=click.Path(exists=True))
@click.option('-j', '--param_json', type=click.Path(exists=True), multiple=True,
              help='algorithm parameters json, possibly from a previous optimise job')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
def optimise(pipeline_file: str, param_json: str, partitions: int) -> None:
    """Optimise model parameters using Bayesian regression."""
    if uncoverml.mpiops.chunks > 1 and ('PBS_NNODES' in os.environ and int(os.environ['PBS_NNODES']) > 1):
        raise NotImplementedError("Currently optimiser does not work with mpi. \n"
                                  "However it can utilise a whole NCI node with many CPUs!")
    config = ls.config.Config(pipeline_file)
    if len(param_json) > 0:
        # log.info('the following jsons are used as params \n', click.echo('\n'.join(param_json)))
        param_dicts = []  # list of dicts
        for pj in param_json:
            with open(pj, 'r') as f:
                log.info(f"{config.algorithm} params were updated using {param_json}")
                param_dicts.append(json.load(f))
        config.algorithm_args = {k: v for D in param_dicts for k, v in D.items()}

    param_str = f'Optimising {config.algorithm} model with the following base params:\n'
    write_progress_to_file('opt', 'Starting optimisation', config)
    for param, value in config.algorithm_args.items():
        param_str += "{}\t= {}\n".format(param, value)
    log.info(param_str)

    write_progress_to_file('opt', 'Loading targets', config)
    targets_all, x_all = _load_data(config, partitions)
    write_progress_to_file('opt', 'Targets loaded', config)

    if ls.mpiops.chunk_index == 0:
        if config.hpopt:
            log.info("Using hyperopt package to optimise model params")
            write_progress_to_file('opt', 'Optimising model params '
                                          'using the hyperopt package', config)
            hyopt.optimise_model(x_all, targets_all, config)
        else:
            log.info("Using scikit-optimise package to optimise model params")
            optimisation.bayesian_optimisation(x_all, targets_all, config)

    ls.mpiops.comm.barrier()
    log.info("Finished! Total mem = {:.1f} GB".format(_total_gb()))

    log.info("Finished optimisation of model parameters!")


@cli.command()
@click.argument('pipeline_file')
@click.option('-s', '--subsample_fraction', type=float, default=1.0,
              help='only use this fraction of the data for learning classes')
def cluster(pipeline_file, subsample_fraction):
    config = ls.config.Config(pipeline_file)

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

    if config.semi_supervised:
        semisupervised(config)
    else:
        unsupervised(config)
    log.info("Finished! Total mem = {:.1f} GB".format(_total_gb()))


def semisupervised(config):

    # make sure we're clear that we're clustering
    config.algorithm = config.clustering_algorithm
    config.cubist = False
    # Get the taregts
    targets = ls.geoio.load_targets(shapefile=config.class_file,
                                    targetfield=config.class_property,
                                    conf=config)

    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.geoio.semisupervised_feature_sets(targets, config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    features, _ = ls.features.transform_features(image_chunk_sets,
                                                 transform_sets,
                                                 config.final_transform,
                                                 config)
    features, classes = ls.features.remove_missing(features, targets)
    indices = np.arange(classes.shape[0], dtype=int)

    config.n_classes = ls.cluster.compute_n_classes(classes, config)
    model = ls.cluster.KMeans(config.n_classes, config.oversample_factor)
    log.info("Clustering image")
    model.learn(features, indices, classes)
    ls.mpiops.run_once(ls.geoio.export_cluster_model, model, config)


def unsupervised(config):
    # make sure we're clear that we're clustering
    config.algorithm = config.clustering_algorithm
    config.cubist = False
    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.geoio.unsupervised_feature_sets(config)
    transform_sets = [k.transform_set for k in config.feature_sets]
    features, _ = ls.features.transform_features(image_chunk_sets,
                                                 transform_sets,
                                                 config.final_transform,
                                                 config)

    features, _ = ls.features.remove_missing(features)
    model = ls.cluster.KMeans(config.n_classes, config.oversample_factor)
    log.info("Clustering image")
    model.learn(features)
    ls.mpiops.run_once(ls.geoio.export_cluster_model, model, config)

    features_save_path = os.path.join(config.output_dir, 'training_data.data')
    joblib.dump(features, features_save_path)


@cli.command()
@click.argument('pipeline_file')
@click.argument('model_or_cluster_file')
@click.argument('calling_process')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
@click.option('-i', '--interface_job', is_flag=True,
              help='Flag that the call is coming from an interface job')
def validate(pipeline_file, model_or_cluster_file, calling_process, partitions, interface_job):
    """Validate a model with out-of-sample shapefile."""
    with open(model_or_cluster_file, 'rb') as f:
        state_dict = joblib.load(f)

    model = state_dict["model"]

    config = ls.config.Config(pipeline_file)
    # config.pickle_load = False
    # config.target_file = config.oos_validation_file
    # config.target_property = config.oos_validation_property
    write_progress_to_file(calling_process, 'Loading targets', config)
    targets_all, x_all = _load_data(config, partitions)
    write_progress_to_file(calling_process, 'Targets loaded', config)

    write_progress_to_file(calling_process, 'Beginning model validation', config)
    ls.validate.oos_validate(targets_all, x_all, model, config, calling_process)
    ls.validate.plot_feature_importance(model, x_all, targets_all, config, calling_process)
    write_progress_to_file(calling_process, 'Model validated', config)

    if interface_job:
        write_progress_to_file(calling_process, 'Uploading files to AWS', config)
        uncoverml.interface_utils.read_presigned_urls_and_upload(config, calling_process)

    write_progress_to_file(calling_process, 'Full Process Complete', config)
    log.info("Finished OOS validation job! Total mem = {:.1f} GB".format(_total_gb()))


@cli.command()
@click.argument('model_or_cluster_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
@click.option('-m', '--mask', type=str, default='',
              help='mask file used to limit prediction area')
@click.option('-r', '--retain', type=int, default=None,
              help='mask values where to predict')
@click.option('-t', '--prediction_template', type=click.Path(exists=True), default=None,
              help='mask values where to predict')
@click.option('-i', '--interface_job', is_flag=True,
              help='Flag that the call is coming from an interface job')
def predict(model_or_cluster_file, partitions, mask, retain, prediction_template, interface_job):
    with open(model_or_cluster_file, 'rb') as f:
        state_dict = joblib.load(f)

    model = state_dict["model"]
    config = state_dict["config"]
    config.is_prediction = True
    config.cluster = True if splitext(model_or_cluster_file)[1] == '.cluster' \
        else False
    config.mask = mask if mask else config.mask
    config.prediction_template = prediction_template if prediction_template else config.prediction_template
    if config.mask:
        config.retain = retain if retain else config.retain

        if not isfile(config.mask):
            config.mask = ''
            log.info('A mask was provided, but the file does not exist on '
                     'disc or is not a file.')

    config.n_subchunks = partitions
    if config.n_subchunks > 1:
        log.info("Memory contstraint forcing {} iterations "
                 "through data".format(config.n_subchunks))
    else:
        log.info("Using memory aggressively: dividing all data between nodes")

    image_shape, image_bbox, image_crs = ls.geoio.get_image_spec(model, config)

    outfile_tif = config.algorithm
    predict_tags = model.get_predict_tags()

    # 30/9 Quick hack to get around issue that I'm having - Adi
    if hasattr(config, 'outbands'):
        if not config.outbands:
            config.outbands = len(predict_tags)
    else:
        config.outbands = len(predict_tags)

    if not hasattr(config, 'geotif_options'):
        config.geotif_options = {}

    if not hasattr(config, 'thumbnails'):
        config.thumbnails = 10

    image_out = ls.geoio.ImageWriter(image_shape, image_bbox, image_crs,
                                     outfile_tif,
                                     config.n_subchunks, config.output_dir,
                                     band_tags=predict_tags[
                                               0: min(len(predict_tags),
                                                      config.outbands)],
                                     **config.geotif_options)

    for i in range(config.n_subchunks):
        log.info("starting to render partition {}".format(i+1))
        ls.predict.render_partition(model, i, image_out, config)
        prediction_pct = float(i) / float(config.n_subchunks)
        write_progress_to_file('pred', f'Prediction: {prediction_pct: .2%} Rendered', config)

    # explicitly close output rasters
    image_out.close()

    if config.cluster and config.cluster_analysis:
        if ls.mpiops.chunk_index == 0:
            ls.predict.final_cluster_analysis(config.n_classes,
                                              config.n_subchunks)

    # ls.predict.final_cluster_analysis(config.n_classes,
    #                                   config.n_subchunks)

    if config.thumbnails:
        image_out.output_thumbnails(config.thumbnails)

    if interface_job:
        write_progress_to_file('pred', 'Preparing results for upload', config)
        uncoverml.interface_utils.rename_files_before_upload(config)
        uncoverml.interface_utils.create_thumbnail('prediction', config)
        uncoverml.interface_utils.calc_std(config)
        uncoverml.interface_utils.create_thumbnail('std', config)
        uncoverml.interface_utils.create_results_zip(config)
        uncoverml.interface_utils.read_presigned_urls_and_upload(config, 'pred')
        write_progress_to_file('pred', 'Uploading results to AWS', config)

    write_progress_to_file('pred', 'Full Process Complete', config)
    log.info("Finished! Total mem = {:.1f} GB".format(_total_gb()))


@cli.command()
@click.argument('pipeline_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
@click.option('-s', '--subsample_fraction', type=float, default=1.0,
              help='only use this fraction of the data for learning classes')
@click.option('-m', '--mask', type=str, default='',
              help='mask file used to limit prediction area')
@click.option('-r', '--retain', type=int, default=None,
              help='mask values where to predict')
def pca(pipeline_file, partitions, subsample_fraction, mask, retain):
    config = ls.config.Config(pipeline_file)

    __validate_pca_config(config)  # no other transforms other than whiten
    assert config.pca, "Not a pca analysis. Please include the pca block in your yaml!!"
    config.mask = mask if mask else config.mask

    config.subsample_fraction = subsample_fraction
    if config.mask:
        config.retain = retain if retain else config.retain

        if not isfile(config.mask):
            config.mask = ''
            log.info('A mask was provided, but the file does not exist on '
                     'disc or is not a file.')

    config.n_subchunks = partitions
    if config.n_subchunks > 1:
        log.info("Memory contstraint forcing {} iterations through data".format(config.n_subchunks))
    else:
        log.info("Using memory aggressively: dividing all data between nodes")

    # Get the image chunks and their associated transforms
    image_chunk_sets = ls.geoio.unsupervised_feature_sets(config)
    transform_sets = [k.transform_set for k in config.feature_sets]

    features, _ = ls.features.transform_features(image_chunk_sets,
                                         transform_sets,
                                         config.final_transform,
                                         config)
    features, _ = ls.features.remove_missing(features)
    num_samples = np.sum(ls.mpiops.comm.gather(features.shape[0]))
    log.info(f"Extracting the top {features.shape[1]} PCs from a random sampling of"
             f" {num_samples} points from the rasters")
    log.info(f"Done whiten tranform with {subsample_fraction*100}% of all data")
    ls.mpiops.run_once(ls.geoio.export_cluster_model, "dummy_model", config)
    ls.mpiops.run_once(ls.predict.export_pca_fractions, config)

    whiten_transform = config.final_transform.global_transforms[0]
    image_shape, image_bbox, image_crs = ls.geoio.get_image_spec_from_nchannels(whiten_transform.keepdims, config)
    outfile_tif = config.name + "_pca"

    image_out = ls.geoio.ImageWriter(image_shape, image_bbox, image_crs,
                                     outfile_tif,
                                     config.n_subchunks, config.output_dir,
                                     band_tags=[f'pc_{n}' for n in range(1, whiten_transform.keepdims+1)],
                                     **config.geotif_options)
    for i in range(config.n_subchunks):
        log.info("starting to render partition {}".format(i+1))
        ls.predict.export_pca(i, image_out, config)

    # explicitly close output rasters
    image_out.close()

    log.info("Finished! Total mem = {:.1f} GB".format(_total_gb()))


def __validate_pca_config(config):
    # assert no other transforms other than whiten
    transform_sets = [k.transform_set for k in config.feature_sets]
    for t in transform_sets:
        assert len(t.image_transforms) == 0  # validation that there are no image or global transforms
        assert len(t.global_transforms) == 0
    assert len(config.final_transform.global_transforms) == 1
    assert isinstance(config.final_transform.global_transforms[0], WhitenTransform)
    return config.final_transform.global_transforms[0]


def _total_gb():
    # given in KB so convert
    my_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    # total_usage = mpiops.comm.reduce(my_usage, root=0)
    total_usage = ls.mpiops.comm.allreduce(my_usage)
    return total_usage


if __name__ == '__main__':
    test_config = '/g/data/jl14/jobs/testjob_20231003152254/config.yaml'
    test_model = '/g/data/jl14/jobs/testjob_20231003152254/results/config_optimised.model'
    validate(test_config, test_model, 'opt', 10)
