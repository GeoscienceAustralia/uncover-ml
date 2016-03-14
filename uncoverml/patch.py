""" Image patch extraction and windowing utilities. """

from __future__ import division

import numpy as np
    
def patches(image, pointspec, patch_width):
    """
    High-level function abstracting over what sort of patches we're getting.

    Parameters
    ----------
        image: ndarray
            an array of shape (x,y or (x,y,channels)
        pointspec: GridPointSpec or ListPointSpec
            a pointspec object describing the required patch locations
        patch_width: int
            the half-width of the square patches to extract

    Returns
    -------
        ndarray
            A flattened patch of shape ((pwidth * 2 + 1)**2 * channels,)
    """
    if hasattr(pointspec, 'coords'):
        p = point_patches(images, pointspec.coords, patch_width)
    else:
        raw_gen = grid_patches(image, patch_width, 1) #stride = 1
        p = (k[0] for k in raw_gen) # throw away pixel centres
    return p



def grid_patches(image, pwidth, pstride, centreoffset=None):
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
        pstride: int
            the stride (in pixels) between successive patches.
        centreoffset: tuple, optional
            a tuple of (x, y) offsets to add to the patch centres (centrex
            and centrey)

    Yields
    ------
        patch: ndarray
            An image patch of shape (psize, psize, channels,), where
            psize = pwidth * 2 + 1
        centrex: float
            the centre (x coords) of the patch.
        centrey: float
            the centre (y coords) of the patch.
    """

    # Check and get image dimensions
    Ih, Iw, Ic = _checkim(image)
    psize = pwidth * 2 + 1
    pstride = max(1, pstride)

    # Extract the patches and get the patch centres
    for x in _spacing(Ih, psize, pstride):           # Rows
        patchx = slice(x, x + psize)

        for y in _spacing(Iw, psize, pstride):       # Cols
            patchy = slice(y, y + psize)

            patch = image[patchx, patchy]
            centrex = x + pwidth
            centrey = y + pwidth

            if centreoffset is not None:
                centrex += centreoffset[0]
                centrey += centreoffset[1]

            yield (patch, centrex, centrey)


def point_patches(image, points, pwidth):
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
        ndarray
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


def image_windows(imshape, nwindows, pwidth, pstride):
    """
    Create sub-windows of an image.

    Parameters
    ----------
        imshape: tuple
            a tuple representing the image shape; (x, y).
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
            two slices (slice_x, slice_y) that represents a subwindow.
    """

    # Get nearest number of windows that preserves aspect ratio
    npside = int(round(np.sqrt(nwindows)))
    psize = pwidth * 2 + 1
    pstride = max(1, pstride)

    # If we split into windows using spacing calculated over the whole image,
    # all the patches etc should be extracted as if they were extracted from
    # one window
    spacex = np.array_split(_spacing(imshape[0], psize, pstride), npside)
    spacey = np.array_split(_spacing(imshape[1], psize, pstride), npside)

    slices = [(slice(sx[0], sx[-1] + psize), slice(sy[0], sy[-1] + psize))
              for sx in spacex if len(sx) > 0
              for sy in spacey if len(sy) > 0]

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
