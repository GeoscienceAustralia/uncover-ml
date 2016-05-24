import numpy as np
import tables as hdf

from uncoverml import geoio
from uncoverml import patch


def output_features(feature_vector, outfile, featname="features",
                    shape=None, bbox=None, targind=None):
    """
    Writes a vector of features out to a standard HDF5 format. The function
    assumes that it is only 1 chunk of a larger vector, so outputs a numerical
    suffix to the file as an index.

    Parameters
    ----------
        feature_vector: array
            A 2D numpy array of shape (nPoints, nDims) of type float. This can
            be a masked array.
        outfile: path
            The name of the output file
        featname: str, optional
            The name of the features.
        shape: tuple, optional
            The original shape of the feature for reproducing an image
        bbox: ndarray, optional
            The bounding box of the original data for reproducing an image
        targind: ndarray, optional
            The indices of the associated target variables
    """
    h5file = hdf.open_file(outfile, mode='w')

    # Make sure we are writing "long" arrays
    if feature_vector.ndim < 2:
        feature_vector = feature_vector[:, np.newaxis]
    array_shape = feature_vector.shape

    filters = hdf.Filters(complevel=5, complib='zlib')

    if np.ma.isMaskedArray(feature_vector):
        fobj = feature_vector.data
        if np.ma.count_masked(feature_vector) == 0:
            fmask = np.zeros(array_shape, dtype=bool)
        else:
            fmask = feature_vector.mask
    else:
        fobj = feature_vector
        fmask = np.zeros(array_shape, dtype=bool)

    h5file.create_carray("/", featname, filters=filters,
                         atom=hdf.Float64Atom(), shape=array_shape, obj=fobj)
    h5file.create_carray("/", "mask", filters=filters,
                         atom=hdf.BoolAtom(), shape=array_shape, obj=fmask)

    if shape is not None:
        h5file.getNode('/' + featname).attrs.shape = shape
        h5file.root.mask.attrs.shape = shape
    if bbox is not None:
        h5file.getNode('/' + featname).attrs.bbox = bbox
        h5file.root.mask.attrs.bbox = bbox
    if targind is not None:
        h5file.create_carray("/", "target_indices", filters=filters,
                             atom=hdf.UIntAtom(), shape=targind.shape,
                             obj=targind.astype(np.uint))

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
        inx = np.logical_and(lonlats[:, 0] > image.xmin,
                             lonlats[:, 0] < image.xmax)
        iny = np.logical_and(lonlats[:, 1] > image.ymin,
                             lonlats[:, 1] < image.ymax)
        valid = np.logical_and(inx, iny)
        targind = np.where(valid)[0]
        # FIXME What if targind is empty?
        valid_lonlats = lonlats[valid]
        pixels = image.lonlat2pix(valid_lonlats)
        patches = patch.point_patches(data, patchsize, pixels)
        patch_mask = patch.point_patches(mask, patchsize, pixels)
    else:
        targind = None
        patches = patch.grid_patches(data, patchsize)
        patch_mask = patch.grid_patches(mask, patchsize)

    patch_data = np.array(list(patches), dtype=data_dtype)
    mask_data = np.array(list(patch_mask), dtype=bool)
    result = np.ma.masked_array(data=patch_data, mask=mask_data)
    return result, targind


def __load_hdf5(infiles):
    data_list = []
    for filename in infiles:
        with hdf.open_file(filename, mode='r') as f:
            data = f.root.features[:]
            mask = f.root.mask[:]
            # FIXME This now needs to load the target indices
            a = np.ma.masked_array(data=data, mask=mask)
            data_list.append(a)
    all_data = np.ma.concatenate(data_list, axis=1)
    return all_data


def load_attributes(filename_dict):
    # Only bother loading the first one as they're all the same for now
    fname = filename_dict[0][0]
    shape = None
    bbox = None
    with hdf.open_file(fname, mode='r') as f:
        if 'shape' in f.root.features.attrs:
            shape = f.root.features.attrs.shape
        if 'bbox' in f.root.features.attrs:
            bbox = f.root.features.attrs.bbox
    return shape, bbox


def load_data(filename_dict, chunk_indices):
    """
    we load references to the data into each node, this function runs
    on the node to actually load the data itself.
    """
    data_dict = {i: __load_hdf5(filename_dict[i]) for i in chunk_indices}
    return data_dict


def load_cvdata(filename_dict, cv_chunks, chunk_indices):

    data_dict = load_data(filename_dict, chunk_indices)
    return {i: d[cv_chunks[i]] for i, d in data_dict.items()}


def load_image_data(image_dict, chunk_indices, patchsize, targets):
    """
    we load references to the data into each node, this function runs
    on the node to actually load the data itself.
    """
    data_dict = {i: patches_from_image(image_dict[i], patchsize, targets)
                 for i in chunk_indices}
    return data_dict

def image_data_vector(image_data):
    """
    image_data : dictionary of ndarrays
    """
    indices = sorted(image_data.keys())
    data_list = []
    for i in indices:
        d = image_data[i]
        data_list.append(d.reshape((d.shape[0],-1)))
    x = np.ma.concatenate(data_list, axis=0)
    if np.ma.count_masked(x) == 0 and np.isscalar(x.mask):
        x.mask = np.zeros_like(x,dtype=bool)

    return x

def data_vector(data_dict):
    indices = sorted(data_dict.keys())
    in_d = [data_dict[i][0] for i in indices]
    x = np.ma.concatenate(in_d, axis=0)
    return x
