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
                help="Output shapefile name, "
                "if not specified a random name is used and the file is saved"
                "in the outdir specified in config file")
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
def cli(pipeline_file, outfile, verbosity):
    uncoverml.mllog.configure(verbosity)
    config = ls.config.Config(pipeline_file)
    config.target_file = resample_shapefile(config, outfile)
