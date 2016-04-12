import os.path
import numpy as np
import numpy.ma as ma
import tables as hdf

from uncoverml import geoio
from uncoverml import patch


def output_features(feature_vector, outfile, featname="features"):
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
    """
    h5file = hdf.open_file(outfile, mode='w')
    array_shape = feature_vector.shape

    filters = hdf.Filters(complevel=5, complib='zlib')

    if ma.isMaskedArray(feature_vector):
        fobj = feature_vector.data
        fmask = feature_vector.mask
    else:
        fobj = feature_vector
        fmask = np.zeros(array_shape, dtype=bool)

    h5file.create_carray("/", featname, filters=filters,
                         atom=hdf.Float64Atom(), shape=array_shape, obj=fobj)
    h5file.create_carray("/", "mask", filters=filters,
                         atom=hdf.BoolAtom(), shape=array_shape, obj=fmask)

    # if ma.isMaskedArray(feature_vector):
    #     h5file.root.[:] = feature_vector.data
    #     h5file.root.mask[:] = feature_vector.mask
    # else:
    #     h5file.root.features[:] = feature_vector
    #     h5file.root.mask[:] = np.zeros(array_shape, dtype=bool)

    h5file.close()


def input_features(infile, featname="features"):
    """
    Reads a vector of features out from a standard HDF5 format. The function
    assumes the file it is reading was written by output_features

    Parameters
    ----------
        infile: path
            The name of the input file
        featname: str, optional
            The name of the features.
    Returns
    -------
        data: array
            a 2d numpy array of shape (npoints, ndims) of type float
        mask: array
            a 2d numpy array of shape (npoints, ndims) of type bool (True for
            missing data)
    """
    with hdf.open_file(infile, mode='r') as f:
        data = f.get_node('/' + featname).read()
        mask = f.root.mask.read()
    return data, mask


def transform(x, x_mask):
    return x.flatten(), x_mask.flatten()


def features_from_image(image, name, transform, patchsize, output_dir,
                        targets=None):
    """
    Applies a transform function to a geotiff and writes the output
    as a feature vector to and HDF5 file.
    """
    # Get the target points if they exist:
    data_and_mask = image.data()
    data = data_and_mask.data
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

    transformed_data = [transform(x,m) for x,m in zip(patches, patch_mask)]
    t_patches, t_mask = tuple(zip(*transformed_data))
    features = np.array(t_patches, dtype=float)
    feature_mask = np.array(t_mask, dtype=bool)
    filename = os.path.join(output_dir,
                            name + "_{}.hdf5".format(image.chunk_idx))
    output_features(features, feature_mask, filename)


def cat_chunks(filename_chunks):

    feats = []
    masks = []
    for i, flist in filename_chunks.items():
        feat = []
        mask = []
        for f in flist:
            f, m = input_features(f)
            feat.append(f)
            mask.append(m)
        feats.append(np.hstack(feat))
        masks.append(np.hstack(mask))

    X = np.vstack(feats)
    M = np.vstack(masks)
    return X, M
