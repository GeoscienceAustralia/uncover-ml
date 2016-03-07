""" Image patch extraction and windowing utilities. """

import numpy as np


def grid_patches(image, psize, pstride, centreoffset=None):
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
        centreoffset: tuple, optional
            a tuple of (row, col) offsets to add to the patch centres (centrey
            and centrex)

    yeilds
    ------
        patches: np.array
            A flattened image patch of shape (psize**2 * channels,).
        centrecol: float
            the centre (column coords) of the patch.
        centrerow: float
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
            centrerow = y + float(psize) / 2 - 0.5
            centrecol = x + float(psize) / 2 - 0.5

            if centreoffset is not None:
                centrerow += centreoffset[0]
                centrecol += centreoffset[1]

            yield (patch, centrecol, centrerow)


def point_patches(image, points, psize):

    # TODO
    pass


def image_windows(imshape, nwindows, psize, pstride):
    """
    Create sub-windows of an image.

    Parameters
    ----------
        imshape: tuple
            a tuple representing the image shape; (rows, cols, ...).
        nwindows: int
            the number of windows to divide the image into. The nearest square
            will actually be used (to preserve aspect ratio).
        psize: int
            the size of the square patches to extract, in pixels.
        pstride: int
            the stride (in pixels) between successive patches.

    Returns
    -------
        slices: list
            a list of length round(sqrt(nwindows))**2 of tuples. Each tuple has
            two slices (slice_rows, slice_cols) that represents a subwindow.
    """

    # Get nearest number of windows that preserves aspect ratio
    npside = int(round(np.sqrt(nwindows)))

    # If we split into windows using spacing calculated over the whole image,
    # all the patches etc should be extracted as if they were extracted from on
    # window
    spacex = np.array_split(_spacing(imshape[1], psize, pstride), npside)
    spacey = np.array_split(_spacing(imshape[0], psize, pstride), npside)

    slices = [(slice(sx[0], sx[-1] + psize), slice(sy[0], sy[-1] + psize))
              for sx in spacex for sy in spacey]

    return slices


def _spacing(dimension, psize, pstride):
    """
    Calculate the patch spacings along a dimension of an image.
    """

    offset = int(np.floor(float((dimension - psize) % pstride) / 2))
    return range(offset, dimension - psize + 1, pstride)
