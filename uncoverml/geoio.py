from __future__ import division

import rasterio
import os.path
import numpy as np
from affine import Affine
import shapefile
import tables
import logging

log = logging.getLogger(__name__)


def file_indices_okay(filenames):

    # get just the name eg /path/to/file_0.hdf5 -> file_0
    basenames = [os.path.splitext(os.path.basename(k))[0] for k in filenames]

    # file_0 -> [file,0]
    base_and_idx = [k.rsplit('.part', 1) for k in basenames]
    bases = set([k[0] for k in base_and_idx])
    log.info("Input file sets: {}".format(set(bases)))

    # check every base has the right indices
    # "[[file,0], [file,1]] -> {file:[0,1]}
    try:
        base_ids = {k: set([int(j[1])
                           for j in base_and_idx if j[0] == k]) for k in bases}
    except:
        log.error("One or more filenames are not in <name>.part<idx>.hdf5"
                  "format")
        # either there are no ints at the end or no underscore
        return False

    # determine the 'correct' number of indices (highest index we see)
    num_ids = np.amax(np.array([max(k) for j, k in base_ids.items()])) + 1
    true_set = set(range(num_ids))
    files_ok = True
    for b, nums in base_ids.items():
        if not nums == true_set:
            files_ok = False
            log.error("feature {} has wrong files. ".format(b))
            missing = true_set.difference(nums)
            if len(missing) > 0:
                log.error("Missing Index: {}".format(missing))
            extra = nums.difference(true_set)
            if len(extra) > 0:
                log.error("Extra Index: {}".format(extra))
    return files_ok


def files_by_chunk(filenames):
    """
    returns a dictionary per-chunk of a *sorted* list of features
    Note: assumes files_indices_ok returned true
    """

    # get just the name eg /path/to/file.part0.hdf5 -> file.part0
    transform = lambda x: os.path.splitext(os.path.basename(x))[0]
    sorted_filenames = sorted(filenames, key=transform)
    basenames = [transform(k) for k in sorted_filenames]
    indices = [int(k.rsplit('.part', 1)[1]) for k in basenames]
    d = {i: [] for i in set(indices)}
    for i, f in zip(indices, sorted_filenames):
        d[i].append(f)
    return d


def _invert_affine(A):

    R = np.array([A[0:2], A[3:5]])
    T = np.array([[A[2], A[5]]]).T

    iR = np.linalg.pinv(R)
    iT = -iR.dot(T)
    iA = np.hstack((iR, iT))

    return Affine(*iA.flatten())


def points_from_shp(filename):
    """
    TODO
    """
    # TODO check the shapefile only contains points
    coords = []
    sf = shapefile.Reader(filename)
    for shape in sf.iterShapes():
        coords.append(list(shape.__geo_interface__['coordinates']))
    label_coords = np.array(coords)
    return label_coords


def points_from_hdf(filename, fieldname=None):
    """
    TODO
    """

    with tables.open_file(filename, mode='r') as f:
        lons = f.root.Longitude.read()
        lats = f.root.Latitude.read()
        if fieldname is not None:
            vals = f.get_node("/" + fieldname).read()

    lonlat = np.hstack((lons, lats))
    return lonlat if fieldname is None else (lonlat, vals)


def points_to_hdf(lonlat, outfile, fieldname=None, fieldvals=None):

    with tables.open_file(outfile, 'w') as f:
        f.create_array("/", "Longitude", obj=lonlat[:, 0][:, np.newaxis])
        f.create_array("/", "Latitude", obj=lonlat[:, 1][:, np.newaxis])
        if fieldname is not None:
            f.create_array("/", fieldname, obj=fieldvals)


def values_from_shp(filename, field):
    """
    TODO
    """

    sf = shapefile.Reader(filename)
    fdict = {f[0]: i for i, f in enumerate(sf.fields[1:])}  # Skip DeletionFlag

    if field not in fdict:
        raise ValueError("Requested field is not in records!")

    vind = fdict[field]
    vals = [r[vind] for r in sf.records()]

    return np.array(vals)


def values_from_hdf(filename, field):

    with tables.open_file(filename, mode='r') as f:
        vals = [v for v in f.root.field]

    return np.array(vals)


def construct_splits(npixels, nchunks, overlap=0):
    # Build the equivalent windowed image
    # plus one for exclusive upper range
    y_arrays = np.array_split(np.arange(npixels + 1), nchunks)
    y_bounds = []
    # construct the overlap
    for s in y_arrays:
        old_min = s[0]
        old_max = s[-1]
        new_min = max(0, old_min - overlap)
        new_max = min(npixels, old_max + overlap)
        y_bounds.append((new_min, new_max))
    return y_bounds


def bounding_box(raster):
    """
    TODO
    """
    T1 = raster.affine

    # No shearing or rotation allowed!!
    if not ((T1[1] == 0) and (T1[3] == 0)):
        raise RuntimeError("Transform to pixel coordinates has rotation "
                           "or shear")

    # the +1 because we want pixel corner 1 beyond the last pixel
    lon_range = T1[2] + np.array([0, raster.width + 1]) * T1[0]
    lat_range = T1[5] + np.array([0, raster.height + 1]) * T1[4]

    lon_range = np.sort(lon_range)
    lat_range = np.sort(lat_range)

    # import IPython; IPython.embed()

    return lon_range, lat_range


class Image:
    def __init__(self, filename, chunk_idx=0, nchunks=1, overlap=0):
        assert chunk_idx >= 0 and chunk_idx < nchunks

        self.chunk_idx = chunk_idx
        self.nchunks = nchunks
        self.filename = filename

        # Get the full image details
        with rasterio.open(self.filename, 'r') as geotiff:
            self._full_xrange, self._full_yrange = bounding_box(geotiff)
            self._full_res = (geotiff.width, geotiff.height, geotiff.count)
            self._nodata_value = geotiff.meta['nodata']
            # we don't support different channels with different dtypes
            for d in geotiff.dtypes[1:]:
                if geotiff.dtypes[0] != d:
                    raise ValueError("No support for multichannel geotiffs "
                                     "with differently typed channels")
            self._dtype = np.dtype(geotiff.dtypes[0])

            # Build the affine transformation for the FULL image
            self.A = geotiff.affine
            self.pixsize_x = abs(self.A[0])
            self.pixsize_y = abs(self.A[4])

            # centered pixels
            self.cA = self.A * Affine.translation(0.5, 0.5)

        # Get bounds of window
        xmin, xmax = (0, self._full_res[0])
        ymin, ymax = construct_splits(self._full_res[1],
                                      nchunks, overlap)[chunk_idx]

        # Calculate the new values for resolution and bounding box
        self._offset = np.array([xmin, ymin])
        self.resolution = (xmax - xmin, ymax - ymin, self._full_res[2])
        chunk_xy = np.array([[xmin, ymin], [xmax + 1, ymax + 1]])
        ((lonmin, lat0), (lonmax, lat1)) = self.pix2latlon(chunk_xy,
                                                           centres=False)
        self.bbox = np.array([[lonmin, lonmax],
                              [min(lat0, lat1), max(lat0, lat1)]])

    def __repr__(self):
        return "<geo.Image({}), chunk {} of {})>".format(self.filename,
                                                         self.chunk_idx,
                                                         self.nchunks)

    def data(self):
        # ((ymin, ymax),(xmin, xmax)) plus one for exlusive upper index
        window = ((self._offset[1], self._offset[1] + self.resolution[1] + 1),
                  (self._offset[0], self._offset[0] + self.resolution[0] + 1))
        with rasterio.open(self.filename, 'r') as geotiff:
            d = geotiff.read(window=window, masked=True)
        d = d[np.newaxis, :, :] if d.ndim == 2 else d
        d = np.ma.transpose(d, [2, 1, 0])  # Transpose and channels at back

        # uniform mask format
        if np.ma.count_masked(d) == 0:
            d = np.ma.masked_array(data=d.data,
                                   mask=np.zeros_like(d.data, dtype=bool))
        return d

    @property
    def nodata_value(self):
        return self._nodata_value

    @property
    def dtype(self):
        return self._dtype

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

    def lonlat2pix(self, lonlat, centres=True):

        iA = ~self.cA if centres else ~self.A
        xy = np.floor([np.array(iA * ll) + 1e-5 for ll in lonlat]).astype(int)

        # Allow 1-pixel slop around boundaries because of numerical inaccuracy
        xy[xy == -1] = 0
        xy[xy[:, 0] == self.resolution[0], 0] = self.resolution[0] - 1
        xy[xy[:, 1] == self.resolution[1], 1] = self.resolution[1] - 1

        # subtract the offset because iA is for full image
        xy -= self._offset[np.newaxis, :]

        assert all(np.logical_and(xy[:, 0] >= 0,
                                  xy[:, 0] < self.resolution[0]))
        assert all(np.logical_and(xy[:, 1] >= 0,
                                  xy[:, 1] < self.resolution[1]))

        return xy

    def pix2latlon(self, xy, centres=True):

        A = self.cA if centres else self.A

        # add the offset because A is for full image
        off_xy = xy + self._offset[np.newaxis, :]
        lonlat = np.array([A * pix for pix in off_xy])

        return lonlat


def bbox2affine(xmax, xmin, ymax, ymin, xres, yres):

    pixsize_x = (xmax - xmin) / (xres + 1)
    pixsize_y = (ymax - ymin) / (yres + 1)

    A = Affine(pixsize_x, 0, xmin,
               0, -pixsize_y, ymax)

    return A, pixsize_x, pixsize_y
