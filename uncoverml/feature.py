import os.path
import numpy as np
import tables as hdf

from uncoverml import geoio
from uncoverml import patch


def output_features(feature_vector, outfile):
    """
    Writes a vector of features out to a standard HDF5 format. The function
    assumes that it is only 1 chunk of a larger vector, so outputs a numerical
    suffix to the file as an index.

    Parameters
    ----------
        feature_vector: array
            A 2D numpy array of shape (nPoints, nDims) of type float.
        outfile: path
            The name of the output file
    """
    h5file = hdf.open_file(outfile, mode='w')
    array_shape = feature_vector.shape

    filters = hdf.Filters(complevel=5, complib='zlib')
    h5file.create_carray("/", "features", filters=filters,
                         atom=hdf.Float64Atom(), shape=array_shape)
    h5file.root.features[:] = feature_vector
    h5file.close()

def input_features(infile):
    """
    Reads a vector of features out from a standard HDF5 format. The function
    assumes the file it is reading was written by output_features

    Parameters
    ----------
        feature_vector: array
            A 2D numpy array of shape (nPoints, nDims) of type float.
        infile: path
            The name of the input file
    Returns
    -------
        data: array
            A 2D numpy array of shape (nPoints, nDims) of type float
    """
    with hdf.open_file(infile, mode='r') as f:
        data = f.root.features[:]
    return data

def transform(x):
    return x.flatten()


def features_from_image(image, name, transform, patchsize, output_dir,
                        targets=None):
    """
    Applies a transform function to a geotiff and writes the output
    as a feature vector to and HDF5 file.
    """
    # Get the target points if they exist:
    data = image.data()
    pixels = None
    if targets is not None:
        lonlats = geoio.points_from_hdf(targets)
        inx = np.logical_and(lonlats[:, 0] >= image.xmin,
                             lonlats[:, 0] < image.xmax)
        iny = np.logical_and(lonlats[:, 1] >= image.ymin,
                             lonlats[:, 1] < image.ymax)
        valid = np.logical_and(inx, iny)
        valid_lonlats = lonlats[valid]
        pixels = image.lonlat2pix(valid_lonlats, centres=True)
        patches = patch.point_patches(data, patchsize, pixels)
    else:
        patches = patch.grid_patches(data, patchsize)

    processed_patches = map(transform, patches)
    features = np.array(list(processed_patches), dtype=float)
    filename = os.path.join(output_dir,
                            name + "_{}.hdf5".format(image.chunk_idx))
    output_features(features, filename)

