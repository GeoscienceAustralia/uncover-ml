import os.path
import numpy as np
import tables as hdf
from uncoverml.celerybase import celery
from uncoverml import io
from uncoverml import patch



def output_features(feature_vector, centres, x_idx, y_idx, outfile):
    """
    Writes a vector of features out to a standard HDF5 format. The function
    assumes that it is only 1 chunk of a larger vector, so outputs a numerical
    suffix to the file as an index.

    Parameters
    ----------
        feature_vector: array
            A 2D numpy array of shape (nPoints, nDims) of type float.
        x_idx: uint
            A non-negative integer represting the x chunk index of this data
        y_idx: uint
            A non-negative integer represting the y chunk index of this data
        outfile: path
            A path to and HDF5 file that doesn't exist. The function output
            will add the chunk indices before the .hdf5 extension. For example
            if outfile is out.hdf5 it will write out_1_3.hdf5 for chunk 1,3
    """
    filename = os.path.splitext(outfile)[0] + \
        "_{}_{}.hdf5".format(x_idx, y_idx)
    h5file = hdf.open_file(filename, mode='w')
    array_shape = feature_vector.shape
    centre_shape = centres.shape

    filters = hdf.Filters(complevel=5, complib='zlib')
    h5file.create_carray("/", "features", filters=filters,
                         atom=hdf.Float64Atom(), shape=array_shape)
    h5file.root.features[:] = feature_vector
    h5file.create_carray("/", "centres", filters=filters,
                         atom=hdf.Int64Atom(), shape=centre_shape)
    h5file.root.centres[:] = centres.astype(int)
    h5file.close()


def transform(x):
    return x.flatten()


@celery.task(name='process_window')
def process_window(x_idx, y_idx, axis_splits, geotiff, pointspec, patchsize,
                   transform, outfile):
    """
    Applies a transform function to a window of a geotiff and writes the output
    as a feature vector to and HDF5 file.

    Parameters
    ----------
        x_idx: uint
            The x index of the geotiff window to process.
            0 <= x_idx < axis_splits
        y_idx: uint
            The y index of the geotiff window to process.
            0 <= y_idx < axis_splits
        axis_splits: uint
            The number of splits per axis chunking. Total chunks is square of
            this number.
        geotiff: path
            The path to the geotiff input file
        pointspec: PointSpec
            The specification defining any subsetting or point extraction
            of the geotiff file
        transform: function
            A function that takes a patch as a 3D numpy array and returns
            a feature vector (1D numpy array)
        outfile: path
            Path specification for the HDF5 output. Note numerical indices
            will be postpended to this path for each chunk file.
            Eg. out.hdf5 will output multiple files like out_0_0.hdf5,
            out_0_1.hdf5 etc.
    """
    # fix stride at 1 for now
    stride = 1
    # open the geotiff
    with io.open_raster(geotiff) as raster:
        res = (raster.width, raster.height)
        slices = patch.image_window(x_idx, y_idx, axis_splits, res,
                                    patchsize, stride)

        img = io.read_raster(raster, slices)

    # Operate on the patches
    offset = (slices[0].start, slices[1].start)
    patches, x, y = zip(*patch.grid_patches(img, patchsize, stride, offset))
    processed_patches = map(transform, patches)
    features = np.array(list(processed_patches), dtype=float)
    centres = np.array((x, y)).T

    # Output the feature to an hdf5 file
    output_features(features, centres, x_idx, y_idx, outfile)
