import click

import uncoverml as ls
from uncoverml.targets import resample_shapefile
from uncoverml import config

@click.command()
@click.argument('pipeline_file')
@click.option('-o', '--outfile', type=click.Path(exists=False), default=None,
                help="Output shapefile name, "
                "if not specified a random name is used and the file is saved"
                "in the outdir specified in config file")
def cli(pipeline_file, outfile):
    config = ls.config.Config(pipeline_file)
    config.target_file = resample_shapefile(config, outfile)



