"""
Output a geotiff from a set of HDF5 chunked features

.. program-output:: exportgeotiff --help
"""
from __future__ import division

import logging
import sys
import os.path
import click as cl
import uncoverml.defaults as df
from functools import partial
from uncoverml import parallel
from uncoverml import geoio
from uncoverml import feature
import matplotlib.pyplot as pl
import numpy as np
import rasterio

log = logging.getLogger(__name__)


def transform(x, rows, x_min, x_max, band, separatebands):
    x = x.reshape((rows, -1, x.shape[1]))
    if band is not None:
        x = x[:, :, band:band + 1]
        x_min = x_min[band]
        x_max = x_max[band]

    images = []
    if x_min is not None and x_max is not None:
        x = np.ma.asarray(x, dtype=float)
        x = ((x - x_min) / (x_max - x_min))
        cmap = pl.cm.inferno
        cmap.set_bad(alpha=0)
        for i in range(x.shape[2]):
            rgba = cmap(x[:, :, i])
            rgba = (rgba * 255.0).astype(np.uint8)
            images.append(rgba)
    else:
        if separatebands:
            for i in range(x.shape[2]):
                images.append(x[:, :, i:i + 1])
        else:
            images.append(x)
    return images


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--rgb', is_flag=True, help="Colormap data to rgb format")
@cl.option('--separatebands', is_flag=True, help="Output each band in a"
           "separate geotiff, --rgb flag automatically does this")
@cl.option('--band', type=int, default=None,
           help="Output only a specific band")
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use",
           default=None)
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('name', type=str, required=True)
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(name, files, rgb, separatebands, band, quiet, ipyprofile, outputdir):
    """
    Output a geotiff from a set of HDF5 chunked features.

    This can optionally be a floating point geotiff, or an RGB geotiff.
    Multi-band geotiffs are also optionally output (RGB output has to be per
    band however).
    """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # build full filenames
    full_filenames = [os.path.abspath(f) for f in files]
    log.debug("Input files: {}".format(full_filenames))

    # verify the files are all present
    files_ok = geoio.file_indices_okay(full_filenames)
    if not files_ok:
        sys.exit(-1)

    # build the images
    filename_dict = geoio.files_by_chunk(full_filenames)
    nchunks = len(filename_dict)

    # Get attribs if they exist
    eff_shape, eff_bbox = feature.load_attributes(filename_dict)

    # affine
    A, _, _ = geoio.bbox2affine(eff_bbox[1, 0], eff_bbox[0, 0],
                                eff_bbox[0, 1], eff_bbox[1, 1], *eff_shape)

    # Define the transform function to build the features
    cluster = parallel.direct_view(ipyprofile, nchunks)

    # Load the data into a dict on each client
    # Note chunk_indices is a global with different value on each node
    for i in range(len(cluster)):
        cluster.push({"filenames": filename_dict[i]}, targets=i)
    cluster.execute("x = geoio.load_and_cat(filenames)")

    # Get the bounds for image output
    x_min = None
    x_max = None
    if rgb is True:
        cluster.execute("x_min = np.ma.min(x,axis=0)")
        cluster.execute("x_max = np.ma.max(x,axis=0)")
        x_min = np.amin(np.array(cluster['x_min']), axis=0)
        x_max = np.amax(np.array(cluster['x_max']), axis=0)

    f = partial(transform, rows=eff_shape[0], x_min=x_min,
                x_max=x_max, band=band, separatebands=separatebands)
    cluster.push({"f": f})
    cluster.execute("images = f(x)")

    # Couple of pieces of information we need here
    firstnode = cluster.client[0]
    firstnode.execute("n_images = len(images)")
    firstnode.execute("dtype = images[0].dtype")
    firstnode.execute("n_bands = images[0].shape[2]")
    n_images = firstnode["n_images"]
    dtype = firstnode["dtype"]
    n_bands = firstnode["n_bands"]  # for each image
    nnodes = len(cluster)

    for img_idx in range(n_images):
        band_num = img_idx if band is None else band
        output_filename = os.path.join(outputdir, name +
                                       "_band{}.tif".format(band_num))

        with rasterio.open(output_filename, 'w', driver='GTiff',
                           width=eff_shape[0], height=eff_shape[1],
                           dtype=dtype, count=n_bands, transform=A) as f:
            ystart = 0
            for node in range(nnodes):
                engine = cluster.client[node]
                data = engine["images[{}]".format(img_idx)]
                data = np.ma.transpose(data, [2, 1, 0])  # untranspose
                yend = ystart + data.shape[1]  # this is Y
                window = ((ystart, yend), (0, eff_shape[0]))
                index_list = list(range(1, n_bands + 1))
                f.write(data, window=window, indexes=index_list)
                ystart = yend

    sys.exit(0)
