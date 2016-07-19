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
from functools import partial
from uncoverml import geoio
import matplotlib.pyplot as pl
import numpy as np
import rasterio
from mpi4py import MPI

log = logging.getLogger(__name__)


def max_axis_0(x, y, dtype):
    s = np.amax(np.array([x, y]), axis=0)
    return s


def min_axis_0(x, y, dtype):
    s = np.amin(np.array([x, y]), axis=0)
    return s

max0_op = MPI.Op.Create(max_axis_0, commute=True)
min0_op = MPI.Op.Create(min_axis_0, commute=True)


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
        cmap = pl.cm.inferno if hasattr(pl.cm, 'inferno') else pl.cm.afmhot
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
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--rgb', is_flag=True, help="Colormap data to rgb format")
@cl.option('--separatebands', is_flag=True, help="Output each band in a"
           "separate geotiff, --rgb flag automatically does this")
@cl.option('--band', type=int, default=None,
           help="Output only a specific band")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('name', type=str, required=True)
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(name, files, rgb, separatebands, band, outputdir):
    """
    Output a geotiff from a set of HDF5 chunked features.

    This can optionally be a floating point geotiff, or an RGB geotiff.
    Multi-band geotiffs are also optionally output (RGB output has to be per
    band however).
    """

    # MPI globals
    comm = MPI.COMM_WORLD
    chunks = comm.Get_size()
    chunk_index = comm.Get_rank()

    # build full filenames
    full_filenames = [os.path.abspath(f) for f in files]
    log.debug("Input files: {}".format(full_filenames))

    # verify the files are all present
    files_ok = geoio.file_indices_okay(full_filenames)
    if not files_ok:
        sys.exit(-1)

    # build the images
    filename_dict = geoio.files_by_chunk(full_filenames)

    # Get attribs if they exist
    eff_shape, eff_bbox = geoio.load_attributes(filename_dict)

    # affine
    A, _, _ = geoio.bbox2affine(eff_bbox[1, 0], eff_bbox[0, 0],
                                eff_bbox[0, 1], eff_bbox[1, 1], *eff_shape)

    x = geoio.load_and_cat(filename_dict[chunk_index])
    if x is None:
        raise RuntimeError("Prediction output cant have nodes without data")

    x_min = None
    x_max = None
    if rgb is True:
        x_min_local = np.ma.min(x, axis=0)
        x_max_local = np.ma.max(x, axis=0)
        x_min = comm.allreduce(x_min_local, op=min0_op)
        x_max = comm.allreduce(x_max_local, op=min0_op)

    f = partial(transform, rows=eff_shape[0], x_min=x_min,
                x_max=x_max, band=band, separatebands=separatebands)

    images = f(x)

    # Couple of pieces of information we need here
    if chunk_index != 0:
        reqs = []
        for img_idx in range(len(images)):
            reqs.append(comm.isend(images[img_idx], dest=0, tag=img_idx))
        for r in reqs:
            r.wait()
    else:
        n_images = len(images)
        dtype = images[0].dtype
        n_bands = images[0].shape[2]

        for img_idx in range(n_images):
            band_num = img_idx if band is None else band
            output_filename = os.path.join(outputdir, name +
                                           "_band{}.tif".format(band_num))

            with rasterio.open(output_filename, 'w', driver='GTiff',
                               width=eff_shape[0], height=eff_shape[1],
                               dtype=dtype, count=n_bands, transform=A) as f:
                ystart = 0
                for node in range(chunks):
                    data = comm.recv(source=node, tag=img_idx) \
                        if node != 0 else images[img_idx]
                    data = np.ma.transpose(data, [2, 1, 0])  # untranspose
                    yend = ystart + data.shape[1]  # this is Y
                    window = ((ystart, yend), (0, eff_shape[0]))
                    index_list = list(range(1, n_bands + 1))
                    f.write(data, window=window, indexes=index_list)
                    ystart = yend
