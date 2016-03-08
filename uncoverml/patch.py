""" Image patch extraction and windowing utilities. """

from __future__ import division

import numpy as np


def grid_patches(image, pwidth, pstride, centreoffset=None):
    """
    Generate (overlapping) patches from an image. This function extracts square
    patches from an image in an overlapping, dense grid.

    Parameters
    ----------
        image: np.array,
            an array of shape (rows, cols) or (rows, cols, channels)
        pwidth: int
            the half-width of the square patches to extract, in pixels. E.g.
            pwidth = 0 gives a 1x1 patch, pwidth = 1 gives a 3x3 patch, pwidth
            = 2 gives a 5x5 patch etc. The formula for calculating the full
            patch width is pwidth * 2 + 1.
        pstride: int
            the stride (in pixels) between successive patches.
        centreoffset: tuple, optional
            a tuple of (row, col) offsets to add to the patch centres (centrey
            and centrex)

    yeilds
    ------
        patches: np.array
            A flattened image patch of shape (psize**2 * channels,), where
            psize = pwidth * 2 + 1
        centrerow: float
            the centre (row coords) of the patch.
        centrecol: float
            the centre (column coords) of the patch.
    """

    # Check and get image dimensions
    Ih, Iw, Ic = _checkim(image)
    psize = pwidth * 2 + 1
    rsize = (psize**2) * Ic
    pstride = max(1, pstride)

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

            yield (patch, centrerow, centrecol)


def point_patches(image, points, pwidth):
    """
    
    """

    raise NotImplementedError("Sorry")

    # Check and get image dimensions
    Ih, Iw, Ic = _checkim(image)

    # Calculate patch width and offsets
    psize = pwidth * 2 + 1

    # Make sure points are within bounds of image taking into account psize
    pwidth = psize / 2
    left = top = pwidth
    bottom = Ih - pwidth
    right = Iw - pwidth
    print(left, top, right, bottom)
    if any(top > points[:, 0]) or any(bottom < points[:, 0]) \
            or any(left > points[:, 1]) or any(right < points[:, 1]):
        raise ValueError("Points are outside of image bounds")

    return

    # for p in points:
    #     l = int(np.floor(p[1] - ptl))
    #     r = int(np.floor(p[1] + pbr))
    #     yield image[slice(np.floor(


def image_windows(imshape, nwindows, pwidth, pstride):
    """
    Create sub-windows of an image.

    Parameters
    ----------
        imshape: tuple
            a tuple representing the image shape; (rows, cols, ...).
        nwindows: int
            the number of windows to divide the image into. The nearest square
            will actually be used (to preserve aspect ratio).
        pwidth: int
            the half-width of the square patches to extract, in pixels. E.g.
            pwidth = 0 gives a 1x1 patch, pwidth = 1 gives a 3x3 patch, pwidth
            = 2 gives a 5x5 patch etc. The formula for calculating the full
            patch width is pwidth * 2 + 1.
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
    psize = pwidth * 2 + 1
    pstride = max(1, pstride)

    # If we split into windows using spacing calculated over the whole image,
    # all the patches etc should be extracted as if they were extracted from
    # one window
    spacex = np.array_split(_spacing(imshape[1], psize, pstride), npside)
    spacey = np.array_split(_spacing(imshape[0], psize, pstride), npside)

    slices = [(slice(sy[0], sy[-1] + psize), slice(sx[0], sx[-1] + psize))
              for sy in spacey if len(sy) > 0 for sx in spacex if len(sx) > 0]

    return slices


def _spacing(dimension, psize, pstride):
    """
    Calculate the patch spacings along a dimension of an image.
    """

    offset = int(np.floor(float((dimension - psize) % pstride) / 2))
    return range(offset, dimension - psize + 1, pstride)


def _checkim(image):
    if image.ndim == 3:
        (Ih, Iw, Ic) = image.shape
    elif image.ndim == 2:
        (Ih, Iw) = image.shape
        Ic = 1
    else:
        raise ValueError('image must be a 2D or 3D array')

    return Ih, Iw, Ic
