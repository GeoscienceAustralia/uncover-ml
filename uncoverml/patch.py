""" Image patch extraction and windowing utilities. """

from __future__ import division

import numpy as np


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
    Ix, Iy, Ic = _checkim(image)
    psize = pwidth * 2 + 1
    pstride = max(1, pstride)

    if centreoffset is None:
        centreoffset = (0, 0)

    # Extract the patches and get the patch centres
    for x in _spacing(Ix, psize, pstride):           # Rows
        patchx = slice(x, x + psize)

        for y in _spacing(Iy, psize, pstride):       # Cols
            patchy = slice(y, y + psize)

            patch = image[patchx, patchy]
            centrex = x + pwidth + centreoffset[0]
            centrey = y + pwidth + centreoffset[1]

            yield (patch, centrex, centrey)


def point_patches(image, points, pwidth, centreoffset=None):
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
    Ix, Iy, Ic = _checkim(image)

    import IPython; IPython.embed(); exit()

    # Make sure points are within bounds of image taking into account psize
    if any(points[:, 1] < pwidth) \
            or any(points[:, 1] >= Iy - pwidth) \
            or any(points[:, 0] < pwidth) \
            or any(points[:, 0] >= Ix - pwidth):
        raise ValueError("Points are outside of image bounds!")

    if centreoffset is None:
        centreoffset = (0, 0)

    return ((image[slice(p[0] - pwidth, p[0] + pwidth + 1),
                   slice(p[1] - pwidth, p[1] + pwidth + 1)],
             p[0] + centreoffset[0], p[1] + centreoffset[1])
            for p in points)


def image_window(x_idx, y_idx, axis_splits, imshape, pwidth, pstride):
    """
    Create sub-windows of an image. given specifications for a chunking scheme
    and an index, returns the pixel slices of the image for the chunk at that
    index.

    Parameters
    ----------
        x_idx: int
            the x-index of the chunk. 0 <= x_idx < axis_splits
        y_idx: int
            the y-index of the chunk. 0 <= y_idx < axis_splits
        axis_splits: int
            the number of splits per axis used to create chunks of data.
            The total number of chunks is the square of this value. Aspect
            ratio of the chunks is approximately equal to original image.
            the number of windows to divide the image into. The nearest square
            will actually be used (to preserve aspect ratio).
        imshape: tuple
            a tuple representing the image shape; (x, y).
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
    # Sane indices?
    assert x_idx >= 0 and x_idx < axis_splits
    assert y_idx >= 0 and y_idx < axis_splits

    # Image big enough?
    assert imshape[0] >= axis_splits

    # Full size and correct stride
    psize = pwidth * 2 + 1
    pstride = max(1, pstride)

    # If we split into windows using spacing calculated over the whole image,
    # all the patches etc should be extracted as if they were extracted from
    # one window
    x_patch_corners = _spacing(imshape[0], psize, pstride)
    y_patch_corners = _spacing(imshape[1], psize, pstride)

    assert len(x_patch_corners) >= axis_splits  # at least 1 patch per chunk
    assert len(y_patch_corners) >= axis_splits  # at least 1 patch per chunk

    spacex = np.array_split(x_patch_corners, axis_splits)
    spacey = np.array_split(y_patch_corners, axis_splits)

    sx = spacex[x_idx]
    sy = spacey[y_idx]

    slices = (slice(sx[0], sx[-1] + psize), slice(sy[0], sy[-1] + psize))
    return slices


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
        (Ix, Iy, Ic) = image.shape
    elif image.ndim == 2:
        (Ix, Iy) = image.shape
        Ic = 1
    else:
        raise ValueError('image must be a 2D or 3D array')

    if (Ix < 1) or (Iy < 1):
        raise ValueError('image must be a 2D or 3D array')

    return Ix, Iy, Ic
