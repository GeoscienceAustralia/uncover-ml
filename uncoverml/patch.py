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
        window: tuple, optional
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


def point_patches(points, psize):

    pass


def image_windows(imshape, nchunks, psize, pstride):

    # Get nearest number of chunks that preserves aspect ratio
    npside = int(round(np.sqrt(nchunks)))

    # If we split into windows using spacing calculated over the whole image,
    # all the patches etc should be extracted as if they were extracted from on
    # window
    spacex = np.array_split(_spacing(imshape[1], psize, pstride), npside)
    spacey = np.array_split(_spacing(imshape[0], psize, pstride), npside)

    slices = [(slice(sx[0], sx[-1] + psize), slice(sy[0], sy[-1] + psize))
              for sx in spacex for sy in spacey]

    return slices


def _spacing(dimension, psize, pstride):

    offset = _strideoffset(dimension, psize, pstride)
    return range(offset, dimension - psize + 1, pstride)


def _strideoffset(dimension, psize, pstride):

    return int(np.floor(float((dimension - psize) % pstride) / 2))
