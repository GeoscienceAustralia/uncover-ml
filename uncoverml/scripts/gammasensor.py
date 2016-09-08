import logging
import os.path
import glob

import click
import numpy as np

from uncoverml import geoio
from uncoverml.image import Image
from uncoverml import filtering
from uncoverml import mpiops
import uncoverml.logging

log = logging.getLogger(__name__)


def write_data(data, name, in_image, outputdir, forward):
    data = data.reshape(-1, data.shape[2])
    tag = "convolved" if forward else "deconvolved"
    nbands = data.shape[1]
    if nbands > 1:
        tags = [tag + "_band{}".format(k+1) for k in range(data.shape[1])]
    else:
        tags = [tag]
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
@click.option('--noise', type=float, default=0.001,
              help='noise coeff for the inverse'
              ' transform. Increasing this will remove missing data artifacts'
              ' at the cost of image sharpness')
@click.option('-v', '--verbosity',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Level of logging')
@click.option('-o', '--outputdir', default='.', help='Location to output file')
def cli(verbosity, geotiff, height, absorption, forward, outputdir, noise):
    uncoverml.logging.configure(verbosity)

    log.info("{} simulating gamma sensor model".format(
        "Forward" if forward else "Backward"))
    if os.path.isdir(geotiff):
        log.info("Globbing directory input for tif files")
        geotiff = os.path.join(geotiff, "*.tif")
    files = glob.glob(geotiff)
    my_files = np.array_split(files, mpiops.chunks)[mpiops.chunk_index]
    for f in my_files:
        name = os.path.basename(f).rsplit(".", 1)[0]
        log.info("Loading {}".format(name))
        image_source = geoio.RasterioImageSource(f)
        image = Image(image_source)
        data = image.data()

        # apply transforms here
        log.info("Computing sensor footprint")
        img_w, img_h, _ = data.shape
        S = filtering.sensor_footprint(img_w, img_h,
                                       image.pixsize_x, image.pixsize_y,
                                       height, absorption)
        # Apply and unapply the filter (mirrored boundary conditions)
        log.info("Applying transform to array of shape {}".format(data.shape))
        if forward:
            t_data = filtering.fwd_filter(data, S)
        else:
            orig_mask = data.mask
            if np.ma.count_masked(data) > 0:
                data = filtering.kernel_impute(data, S)
            t_data = filtering.inv_filter(data, S, noise=noise)
            t_data = np.ma.MaskedArray(data=t_data.data, mask=orig_mask)

        # Write output:
        log.info("Writing output to disk")
        write_data(t_data, name, image, outputdir, forward)
    log.info("All files transformed successfully")
