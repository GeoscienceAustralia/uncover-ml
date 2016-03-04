""" Image patch extraction and windowing utilities. """

import numpy as np


def grid_patches(image, psize, pstride):
    """
    Generate (overlapping) patches from an image. This function extracts square
    patches from an image in an overlapping, dense grid.

    Parameters
    ----------
        image: np.array,
            an array of shape (rows, cols) or (rows, cols, channels)
        psize: int
            the size of the square patches to extract, in pixels.
        pstride: int
            the stride (in pixels) between successive patches.

    yeilds
    ------
        patches: np.array
            A flattened image patch of shape (psize**2 * channels,).
        centrex: float
            the centre (column coords) of the patch.
        centrey: float
            the centre (row coords) of the patch.
    """

    # Check and get image dimensions
    if image.ndim == 3:
        (Ih, Iw, Ic) = image.shape
    elif image.ndim == 2:
        (Ih, Iw) = image.shape
        Ic = 1
    else:
        raise ValueError('image must be a 2D or 3D array')

    rsize = (psize**2) * Ic

    # Extract the patches and get the patch centres
    for y in _spacing(Ih, psize, pstride):           # Rows
        patchy = slice(y, y + psize)

        for x in _spacing(Iw, psize, pstride):       # Cols
            patchx = slice(x, x + psize)

            patch = np.reshape(image[patchy, patchx], rsize)
            centrey = y + float(psize) / 2 - 0.5
            centrex = x + float(psize) / 2 - 0.5

            yield (patch, centrex, centrey)


def image_windows(imshape, nchunks, psize, pstride):

    # TODO

    pass


def _spacing(dimension, psize, pstride):

    offset = int(np.floor(float((dimension - psize) % pstride) / 2))
    return range(offset, dimension - psize + 1, pstride)
