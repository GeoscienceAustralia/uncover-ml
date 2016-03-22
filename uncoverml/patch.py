""" Image patch extraction and windowing utilities. """

from __future__ import division

import numpy as np


def grid_patches(image, pwidth, pstride=1):
    """
    Generate (overlapping) patches from an image. This function extracts square
    patches from an image in an overlapping, dense grid.

    Parameters
    ----------
        image: ndarray
            an array of shape (x, y) or (x, y, channels).
        pwidth: int
            the half-width of the square patches to extract, in pixels. E.g.
            pwidth = 0 gives a 1x1 patch, pwidth = 1 gives a 3x3 patch, pwidth
            = 2 gives a 5x5 patch etc. The formula for calculating the full
            patch width is pwidth * 2 + 1.
        pstride: int, optional
            the stride (in pixels) between successive patches.
    Yields
    ------
        patch: ndarray
            An image patch of shape (psize, psize, channels,), where
            psize = pwidth * 2 + 1
    """
    # Check and get image dimensions
    Ih, Iw, Ic = _checkim(image)
    psize = pwidth * 2 + 1

    # Extract the patches and get the patch centres
    for x in _spacing(Ih, psize, pstride):           # Rows
        patchx = slice(x, x + psize)

        for y in _spacing(Iw, psize, pstride):       # Cols
            patchy = slice(y, y + psize)
            patch = image[patchx, patchy]
            yield patch


def point_patches(image, pwidth, points):
    """
    Extract patches from an image at specified points.

    Parameters
    ----------
        image: ndarray
            an array of shape (x, y) or (x, y, channels).
        points: ndarray
           of shape (N, 2) where there are N points, each with an x and y
           coordinate of the patch centre within the image.
        pwidth: int
            the half-width of the square patches to extract, in pixels. E.g.
            pwidth = 0 gives a 1x1 patch, pwidth = 1 gives a 3x3 patch, pwidth
            = 2 gives a 5x5 patch etc. The formula for calculating the full
            patch width is pwidth * 2 + 1.

    Yields
    ------
        patch: ndarray
            An image patch of shape (psize, psize, channels,), where
            psize = pwidth * 2 + 1
    """

    # Check and get image dimensions
    Ih, Iw, Ic = _checkim(image)

    # Make sure points are within bounds of image taking into account psize
    left = top = pwidth
    bottom = Ih - pwidth
    right = Iw - pwidth

    if any(top > points[:, 0]) or any(bottom < points[:, 0]) \
            or any(left > points[:, 1]) or any(right < points[:, 1]):
        raise ValueError("Points are outside of image bounds")

    return (image[slice(p[0] - pwidth, p[0] + pwidth + 1),
                  slice(p[1] - pwidth, p[1] + pwidth + 1)]
            for p in points)


def _spacing(dimension, psize, pstride):
    """
    Calculate the patch spacings along a dimension of an image.
    Returns the lowest-index corner of the patches for a given
    dimension,  size and stride. Always returns at least 1 patch index
    """
    assert dimension >= psize  # otherwise a single patch won't fit
    assert psize > 0
    assert pstride > 0  # otherwise we'll never move along the image

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

    if (Ih < 1) or (Iw < 1):
        raise ValueError('image must be a 2D or 3D array')

    return Ih, Iw, Ic
