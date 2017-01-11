import click
import logging
import uncoverml as ls
from uncoverml.resampling import resample_shapefile
from uncoverml import config
import uncoverml.mllog
log = logging.getLogger(__name__)


@click.command()
@click.argument('pipeline_file')
@click.option('-o', '--outfile', type=click.Path(exists=False), default=None,
                help="Sampled output shapefile name, "
                     "if not specified a random name is used and the file "
                     "is saved in the outdir specified in config file")
@click.option('-s', '--validation_file', type=click.Path(exists=False),
              default=None,
              help="Validation shapefile name, "
                   "if specified a validation shapefile is produced which "
                   "can be used for model validation")
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def cli(pipeline_file, outfile, validation_file, verbosity):
    uncoverml.mllog.configure(verbosity)
    config = ls.config.Config(pipeline_file)
    resample_shapefile(config, outfile, validation_file)
