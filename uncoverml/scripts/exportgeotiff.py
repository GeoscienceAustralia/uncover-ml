"""
Output a geotiff from a set of HDF5 chunked features

.. program-output:: exportgeotiff --help
"""
from __future__ import division

import logging
import sys
import os.path

import click as cl
import click_log as cl_log

from uncoverml import mpiops
from uncoverml import geoio
from uncoverml import image

log = logging.getLogger(__name__)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--rgb', is_flag=True, help="Colormap data to rgb format")
@cl.option('--separatebands', is_flag=True, help="Output each band in a"
           "separate geotiff, --rgb flag automatically does this")
@cl.option('--band', type=int, default=None,
           help="Output only a specific band")
@cl.option('--imagelike', type=cl.Path(exists=True))
@cl.option('--patchsize', type=int, default=0)
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('name', type=str, required=True)
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(name, files, rgb, separatebands, band, outputdir, imagelike,
         patchsize):
    """
    Output a geotiff from a set of HDF5 chunked features.

    This can optionally be a floating point geotiff, or an RGB geotiff.
    Multi-band geotiffs are also optionally output (RGB output has to be per
    band however).
    """

    # build full filenames
    full_filenames = [os.path.abspath(f) for f in files]
    log.debug("Input files: {}".format(full_filenames))

    # verify the files are all present
    files_ok = geoio.file_indices_okay(full_filenames)
    if not files_ok:
        sys.exit(-1)

    # build the images
    filename_dict = geoio.files_by_chunk(full_filenames)
    x = geoio.load_and_cat(filename_dict[mpiops.chunk_index])
    if x is None:
        raise RuntimeError("Prediction output cant have nodes without data")

    # get the details of the image size etc
    template_image = image.Image(geoio.RasterioImageSource(imagelike))
    eff_shape = template_image.patched_shape(patchsize)
    eff_bbox = template_image.patched_bbox(patchsize)
   
    geoio.create_image(x, eff_shape, eff_bbox, name, outputdir, rgb,
                       separatebands, band)
