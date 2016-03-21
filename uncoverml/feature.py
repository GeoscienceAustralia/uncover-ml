import os.path
import numpy as np
import tables as hdf
from uncoverml.celerybase import celery
from uncoverml import io
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


def transform(x):
    return x.flatten()

@celery.task(name='features_from_image')
def features_from_image(name, image, transform, patchsize, targets=None):
    """
    Applies a transform function to a geotiff and writes the output
    as a feature vector to and HDF5 file.
    """
    # Get the target points if they exist:
    pixels = None
    if targets is not None:
        lonlats = geom.points_from_shp(shapefile)
        inx = lonlats[:,0] >= image.xmin and lonlats[:,0] < image.xmax
        iny = lonlats[:,1] >= image.ymin and lonlats[:,1] < image.ymax
        valid = np.logical_and(inx, iny)
        valid_lonlats = lonlats[valid]
        pixels = image.lonlat2pix(lonlats, centres=True)

    data = image.data()
    patches = patch.patches(data, patchsize, pixels)
    processed_patches = map(transform, patches)
    features = np.array(list(processed_patches), dtype=float)
    output_features(features, centres, x_idx, y_idx, name)

