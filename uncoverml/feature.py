import os.path
import numpy as np
import tables as hdf

from uncoverml import geoio
from uncoverml import patch


def output_features(feature_vector, mask_vector, outfile):
    """
    Writes a vector of features out to a standard HDF5 format. The function
    assumes that it is only 1 chunk of a larger vector, so outputs a numerical
    suffix to the file as an index.

    Parameters
    ----------
        feature_vector: array
            A 2D numpy array of shape (nPoints, nDims) of type float.
        mask_vector: array
            A 2D numpy mask array of shape (nPoints, nDims) of type bool
        outfile: path
            The name of the output file
    """
    h5file = hdf.open_file(outfile, mode='w')
    array_shape = feature_vector.shape

    filters = hdf.Filters(complevel=5, complib='zlib')
    h5file.create_carray("/", "features", filters=filters,
                         atom=hdf.Float64Atom(), shape=array_shape)
    h5file.root.features[:] = feature_vector
    h5file.create_carray("/","mask",filters=filters,
                         atom=hdf.BoolAtom(), shape=array_shape)
    h5file.root.mask[:] = mask_vector
    h5file.close()

def patches_from_image(image, patchsize, targets=None):
    """
    Pulls out masked patches from a geotiff, either everywhere or 
    at locations specificed by a targets shapefile
    """
    # Get the target points if they exist:
    data_and_mask = image.data()
    data = data_and_mask.data
    data_dtype = data.dtype
    mask = data_and_mask.mask
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
        patch_mask = patch.point_patches(mask, patchsize, pixels)
    else:
        patches = patch.grid_patches(data, patchsize)
        patch_mask = patch.grid_patches(mask, patchsize)

    patch_data = np.array(list(patches), dtype=data_dtype)
    mask_data = np.array(list(patch_mask), dtype=bool)

    return patch_data, mask_data

