"""
Contains class and routines for reading chunked portions of images.
"""
import numpy as np
import logging

from affine import Affine

_logger = logging.getLogger(__name__)


def construct_splits(npixels, nchunks, overlap=0):
    """
    Splits the image horizontally into approximately equal strips
    according to npixels / nchunks.
    """
    # Build the equivalent windowed image
    # y bounds are EXCLUSIVE
    y_arrays = np.array_split(np.arange(npixels), nchunks)
    y_bounds = []
    # construct the overlap
    for i, s in enumerate(y_arrays):
        if i == 0:
            p_min = s[0]
            p_max = s[-1] + overlap + 1
        elif i == len(y_arrays) - 1:
            p_min = s[0] - overlap
            p_max = s[-1] + 1
        else:
            p_min = s[0] - overlap
            p_max = s[-1] + overlap + 1
        y_bounds.append((p_min, p_max))
    return y_bounds


class Image:
    """
    Represents a raster Image. Can use to get a georeferenced chunk
    of an Image and the data associated with it. This class is mainly
    used in the :mod:`~uncoverml.features` module for intersecting 
    image chunks with target data and extracting the image data. It's
    also used in :mod:`~uncoverml.geoio` for getting covariate specs,
    such as CRS and bounds.

    If nchunks > 1, then the Image is striped horizontally.
    Chunk_idx 0 is the first strip of the image. The X range covers 
    the full width of the image and the Y ranges from 0 to image_height
    / n_chunks.
    
    Parameters
    ----------
    source : :class:`~uncoverml.geoio.ImageSource`
        An instance of ImageSource (typically RasterioImageSource).
        Defines the image to be loaded.
    chunk_idx : int
        Which chunk of the image is being loaded.
    nchunks : int
        Total number of chunks being used. This is typically set
        by the `partitions` parameter of the top level command, 
        also set as `n_subchunks` on the Config object.
    overlap : int
        Doesn't seem to be used, but appears to be used for accomodating
        overlap in chunks (number of rows to overlap with bounding 
        strips).
    """
    def __init__(self, source, chunk_idx=0, nchunks=1, overlap=0):
        assert chunk_idx >= 0 and chunk_idx < nchunks

        if nchunks == 1 and overlap != 0:
            _logger.warning("Ignoring overlap when 1 chunk present")
            overlap = 0

        self.chunk_idx = chunk_idx
        self.nchunks = nchunks
        self.source = source

        _logger.debug("Image has resolution {}".format(source.full_resolution))
        _logger.debug("Image has datatype {}".format(source.dtype))
        _logger.debug("Image missing value: {}".format(source.nodata_value))

        self._full_res = source.full_resolution
        self._start_lon = source.origin_longitude
        self._start_lat = source.origin_latitude
        self.pixsize_x = source.pixsize_x
        self.pixsize_y = source.pixsize_y
        self.crs = source.crs
        assert self.pixsize_x > 0
        assert self.pixsize_y > 0

        # construct the canonical pixel<->position map
        pix_x = range(self._full_res[0] + 1)  # outer corner of last pixel
        coords_x = [self._start_lon + float(k) * self.pixsize_x
                    for k in pix_x]
        self._coords_x = coords_x
        pix_y = range(self._full_res[1] + 1)  # ditto
        coords_y = [self._start_lat + float(k) * self.pixsize_y
                    for k in pix_y]
        self._coords_y = coords_y
        self._pix_x_to_coords = dict(zip(pix_x, coords_x))
        self._pix_y_to_coords = dict(zip(pix_y, coords_y))

        # exclusive y range of this chunk in full image
        ymin, ymax = construct_splits(self._full_res[1],
                                      nchunks, overlap)[chunk_idx]
        self._offset = np.array([0, ymin], dtype=int)
        # exclusive x range of this chunk (same for all chunks)
        xmin, xmax = 0, self._full_res[0]

        assert(xmin < xmax)
        assert(ymin < ymax)

        # get resolution of this chunk
        xres = self._full_res[0]
        yres = ymax - ymin

        # Calculate the new values for resolution and bounding box
        self.resolution = (xres, yres, self._full_res[2])

        start_bound_x, start_bound_y = self._global_pix2lonlat(
            np.array([[xmin, ymin]]))[0]
        # one past the last pixel
        outer_bound_x, outer_bound_y = self._global_pix2lonlat(
            np.array([[xmax, ymax]]))[0]
        assert(start_bound_x < outer_bound_x)
        assert(start_bound_y < outer_bound_y)
        self.bbox = [[start_bound_x, outer_bound_x],
                     [start_bound_y, outer_bound_y]]

    def __repr__(self):
        return "<geo.Image({}), chunk {} of {})>".format(self.source,
                                                         self.chunk_idx,
                                                         self.nchunks)

    def data(self):
        xmin = self._offset[0]
        xmax = self._offset[0] + self.resolution[0]
        ymin = self._offset[1]
        ymax = self._offset[1] + self.resolution[1]
        data = self.source.data(xmin, xmax, ymin, ymax)
        return data

    @property
    def nodata_value(self):
        return self.source.nodata_value

    @property
    def dtype(self):
        return self.source.dtype

    @property
    def xres(self):
        return self.resolution[0]

    @property
    def yres(self):
        return self.resolution[1]

    @property
    def channels(self):
        return self.resolution[2]

    @property
    def npoints(self):
        return self.resolution[0] * self.resolution[1]

    @property
    def x_range(self):
        return self.bbox[0]

    @property
    def y_range(self):
        return self.bbox[1]

    @property
    def xmin(self):
        return self.bbox[0][0]

    @property
    def xmax(self):
        return self.bbox[0][1]

    @property
    def ymin(self):
        return self.bbox[1][0]

    @property
    def ymax(self):
        return self.bbox[1][1]

    def patched_shape(self, patchsize):
        eff_shape = (self.xres - 2 * patchsize,
                     self.yres - 2 * patchsize)
        return eff_shape

    def patched_bbox(self, patchsize):
        start = [patchsize, patchsize]
        end_p1 = [self.xres - patchsize,
                  self.yres - patchsize]
        xy = np.array([start, end_p1])
        eff_bbox = self.pix2lonlat(xy)
        return eff_bbox

    def _global_pix2lonlat(self, xy):
        result = np.array([[self._pix_x_to_coords[x],
                           self._pix_y_to_coords[y]] for x, y in xy])
        return result

    def pix2lonlat(self, xy):
        result = self._global_pix2lonlat(xy + self._offset)
        return result

    def _global_lonlat2pix(self, lonlat):
        x = np.searchsorted(self._coords_x, lonlat[:, 0], side='right') - 1
        x = x.astype(int)
        ycoords = self._coords_y
        y = np.searchsorted(ycoords, lonlat[:, 1], side='right') - 1
        y = y.astype(int)

        # We want the *closed* interval, which means moving
        # points on the end back by 1
        on_end_x = lonlat[:, 0] == self._coords_x[-1]
        on_end_y = lonlat[:, 1] == self._coords_y[-1]
        x[on_end_x] -= 1
        y[on_end_y] -= 1
        if (not all(np.logical_and(x >= 0, x < self._full_res[0]))) or \
                (not all(np.logical_and(y >= 0, y < self._full_res[1]))):
            raise ValueError("Queried location is not "
                             "in the image {}!".format(self.source._filename))

        result = np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)
        return result

    def lonlat2pix(self, lonlat):
        result = self._global_lonlat2pix(lonlat) - self._offset
        # check the postcondition
        x = result[:, 0]
        y = result[:, 1]

        if (not all(np.logical_and(x >= 0, x < self.resolution[0]))) or \
                (not all(np.logical_and(y >= 0, y < self.resolution[1]))):

            raise ValueError("Queried location is not in the image!")

        return result

    def in_bounds(self, lonlat):
        xy = self._global_lonlat2pix(lonlat)
        xy -= self._offset
        x = xy[:, 0]
        y = xy[:, 1]
        inx = np.logical_and(x >= 0, x < self.resolution[0])
        iny = np.logical_and(y >= 0, y < self.resolution[1])
        result = np.logical_and(inx, iny)
        return result


def bbox2affine(xmax, xmin, ymax, ymin, xres, yres):

    pixsize_x = (xmax - xmin) / xres
    pixsize_y = (ymax - ymin) / yres

    A = Affine(pixsize_x, 0, xmin,
               0, -pixsize_y, ymax)

    return A, pixsize_x, pixsize_y

