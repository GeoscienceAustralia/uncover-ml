"""
Output a geotiff from a set of HDF5 chunked features

.. program-output:: exportgeotiff --help
"""
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


def transform(x, rows):
    x = x.reshape((rows, -1, x.shape[1]))
    return x


def colormap(x, x_min, x_max):
    x = np.ma.asarray(x, dtype=float)
    x = ((x - x_min) / (x_max - x_min))
    cmap = pl.cm.viridis
    cmap.set_bad(alpha=0)
    channel_data = []
    for i in range(x.shape[2]):
        rgba = cmap(x[:, :, i])
        rgba = (rgba * 255.0).astype(np.uint8)
        channel_data.append(rgba)
    return channel_data


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--rgb', is_flag=True, help="Colormap data to rgb format")
@cl.option('--band', type=int, default=None,
           help="Output only a specific band")
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use",
           default=None)
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('name', type=str, required=True)
@cl.argument('files', type=cl.Path(exists=True), nargs=-1)
def main(name, files, rgb, band, quiet, ipyprofile, outputdir):
    """ TODO
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

    # Define the transform function to build the features
    cluster = parallel.direct_view(ipyprofile, nchunks)

    # Load the data into a dict on each client
    # Note chunk_indices is a global with different value on each node
    cluster.push({"filename_dict": filename_dict})
    cluster.execute("data_dict = feature.load_data("
                    "filename_dict, chunk_indices)")
    cluster.execute("x = feature.data_vector(data_dict)")

    # Get the bounds for image output
    x_min = None
    x_max = None
    cluster.execute("x_min = np.ma.min(x,axis=0)")
    cluster.execute("x_max = np.ma.max(x,axis=0)")
    x_min = np.amin(np.array(cluster['x_min']), axis=0)
    x_max = np.amax(np.array(cluster['x_max']), axis=0)
    g = partial(colormap, x_min=x_min, x_max=x_max)

    f = partial(transform, rows=eff_shape[0])
    cluster.push({"f": f})
    cluster.execute("t_data_dict = {i: f(k) for i, k in data_dict.items()}")
    cluster.push({"g": g})
    cluster.execute("c_data_dict = {i: g(k) for i, k in t_data_dict.items()}")
   

    output_dtype = np.uint8
    output_count = 4
    output_filename = os.path.join(outputdir, name + ".tif")

    nnodes = len(cluster)
    indices = np.array_split(np.arange(nchunks), nnodes)  # canonical

    with rasterio.open(output_filename, 'w', driver='GTiff',
                       width=eff_shape[0], height=eff_shape[1],
                       dtype=output_dtype, count=output_count) as f:
        ystart = 0
        for node in range(nnodes):
            engine = cluster.client[node]
            data_dict = engine["c_data_dict"]
            node_indices = indices[node]
            for i in node_indices:
                data = data_dict[i][0]
                data = np.ma.transpose(data, [2, 1, 0])  # untranspose
                yend = ystart + data.shape[1]  # this is Y
                window = ((ystart, yend), (0, eff_shape[0]))
                f.write(data, window=window, indexes=[1, 2, 3, 4])
                ystart = yend

    sys.exit(0)
