import logging
import os.path

import click

from uncoverml import geoio
from uncoverml.image import Image
import uncoverml.logging

log = logging.getLogger(__name__)


def write_data(data, name, in_image, outputdir, forward):
    data = data.reshape(-1, data.shape[2])
    tags = ["convolved"] if forward else ["deconvolved"]
    n_subchunks = 1
    nchannels = in_image.resolution[2]
    eff_shape = in_image.patched_shape(0) + (nchannels,)
    eff_bbox = in_image.patched_bbox(0)
    writer = geoio.ImageWriter(eff_shape, eff_bbox, name,
                               n_subchunks, outputdir, band_tags=tags)
    writer.write(data, 0)


@click.command()
@click.argument('geotiff')
@click.option('--invert', 'forward', flag_value=False,
              help='Apply inverse sensor model')
@click.option('--apply', 'forward', flag_value=True, default=True,
              help='Apply forward sensor model')
@click.option('--height', type=float, help='height of sensor')
@click.option('--absorption', type=float, help='absorption coeff')
@click.option('--gain', type=float, help='sensor gain')
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
@click.option('-o', '--outputdir', default='.', help='Location to output file')
def cli(verbosity, geotiff, height, absorption, gain, forward, outputdir):
    uncoverml.logging.configure(verbosity)
    name = os.path.basename(geotiff).rsplit(".", 1)[0]
    image_source = geoio.RasterioImageSource(geotiff)
    image = Image(image_source)
    data = image.data()

    # apply transforms here
    t_data = data * 2

    write_data(t_data, name, image, outputdir, forward)
