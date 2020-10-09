"""
Run the uncoverml pipeline for clustering, supervised learning and prediction.

.. program-output:: uncoverml --help
"""
import platform
if not platform.system() == "Windows":
    import resource
import logging
import pickle
from os.path import isfile, splitext, exists
import os
import shutil
import warnings

import click

import uncoverml
import uncoverml.config
import uncoverml.features
import uncoverml.geoio
import uncoverml.learn
import uncoverml.mllog
import uncoverml.mpiops
import uncoverml.predict
import uncoverml.validate
import uncoverml.targets
import uncoverml.models
from uncoverml.transforms import StandardiseTransform
from uncoverml.scripts import (
    cluster_cli, covdiag_cli, gammasensor_cli, gridsearch_cli, 
    learn_cli, predict_cli, resample_cli, shiftmap_cli, subsample_cli, 
    tiff2kmz_cli, targetsearch_cli, modelfix_cli
)
                               

_logger = logging.getLogger(__name__)
# warnings.showwarning = warn_with_traceback
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


@click.group()
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def cli(verbosity):
    uncoverml.mllog.configure(verbosity)

@cli.command()
@click.argument('config_file')
def modelfix(config_file):
    modelfix_cli.main(config_file)

@cli.command()
@click.argument('config_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
def targetsearch(config_file, partitions):
    targetsearch_cli.main(config_file, partitions)


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('-csv', '--csvfile', default='covdiag.csv', type=click.Path(),
              show_default=True, help='Name of file to store output in CSV format.')
@click.option('-r', 'recursive', is_flag=True, 
              help='Process directories recursively.')
def covdiag(path, csvfile, recursive):
    covdiag_cli.main(path, csvfile, recursive)


@cli.command()
@click.argument('geotiff')
@click.option('--invert', 'forward', flag_value=False,
              help='Apply inverse sensor model')
@click.option('--apply', 'forward', flag_value=True, default=True,
              help='Apply forward sensor model')
@click.option('--height', type=float, required=True, help='height of sensor')
@click.option('--absorption', type=float,  required=True,
              help='absorption coeff')
@click.option('--impute', is_flag=True, help='Use the sensor model to impute'
              ' missing values in the deconvolution')
@click.option('--noise', type=float, default=0.001,
              help='noise coeff for the inverse'
              ' transform. Increasing this will remove missing data artifacts'
              ' at the cost of image sharpness')
@click.option('-o', '--outputdir', default='.', help='Location to output file')
def gammasensor(geotiff, height, absorption, forward, outputdir, noise, impute):
    gammasensor_cli.main(geotiff, height, absorption, forward, outputdir, noise, impute)


@cli.command()
@click.argument('pipeline_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
@click.option('-n', '--njobs', type=int, default=-1,
              help='Number of parallel jobs to run. Lower value of n '
                   'reduces memory requirement. '
                   'By default uses all available CPUs')
def gridsearch(pipeline_file, partitions, njobs):
    gridsearch_cli.main(pipeline_file, partitions, njobs)


@cli.command()
@click.argument('filename')
@click.argument('outputdir')
@click.option('-n', '--npoints', type=int, default=1000,
              help='Number of points to keep')
def subsample(filename, outputdir, npoints):
    subsample_cli.main(filename, outputdir, npoints)


@cli.command()
@click.argument('config_file')
def resample(config_file):
    resample_cli.main(config_file)


@cli.command()
@click.argument('tiff', type=click.Path(exists=True))
@click.option('--outfile', type=click.Path(exists=False), default=None,
        help="Output filename, if not specified then input filename is used")
@click.option('--overlayname', type=str, default=None)
def tiff2kmz(tiff, outfile, overlayname):
    tiff2kmz_cli.main(tiff, outfile, overlayname)


@cli.command()
@click.argument('config_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
def shiftmap(config_file, partitions):
    shiftmap_cli.main(config_file, partitions)


@cli.command()
@click.argument('config_file')
@click.option('-p', '--partitions', type=int, default=1,
              help='divide each node\'s data into this many partitions')
def learn(config_file, partitions):
    learn_cli.main(config_file, partitions)


@cli.command()
@click.argument('config_file')
@click.option('-s', '--subsample_fraction', type=float, default=1.0,
              help="only use this fraction of the data for learning classes")
def cluster(config_file, subsample_fraction):
    cluster_cli.main(config_file, subsample_fraction)


@cli.command()
@click.argument('config_file')
@click.option('-p', '--partitions', type=int, default=1,
              help="divide each node\'s data into this many partitions")
@click.option('-m', '--mask', type=str, default='',
              help="mask file used to limit prediction area")
@click.option('-r', '--retain', type=int, default=None,
              help="mask values where to predict")
def predict(config_file, partitions, mask, retain):
    predict_cli.main(config_file, partitions, mask, retain)


def total_gb():
    if platform.system() == 'Windows':
        _logger.info("Resource usage not yet implemented on Windows. Setting memory usage as 0.")
        total_usage = 0.0
    else:
        # given in KB so convert
        s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        total_usage = uncoverml.mpiops.comm_world.allreduce(s)
    return total_usage


def _clean_temp_cropfiles(config):
    shutil.rmtree(config.tmpdir)   

