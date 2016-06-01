from __future__ import division

from contracts import contract
import rasterio
import os.path
import numpy as np
from affine import Affine
import shapefile
import tables as hdf
import logging

log = logging.getLogger(__name__)


def flatten_image(im):
    x = im.reshape(im.shape[0], -1)
    return x


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
    def transform(x):
        return os.path.splitext(os.path.basename(x))[0]
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

    with hdf.open_file(filename, mode='r') as f:
        lons = f.root.Longitude.read()
        lats = f.root.Latitude.read()
        if fieldname is not None:
            vals = f.get_node("/" + fieldname).read()

    lonlat = np.hstack((lons, lats))
    return lonlat if fieldname is None else (lonlat, vals)


def indices_from_hdf(filename):
    """
    TODO
    """
    with hdf.open_file(filename, mode='r') as f:
        indices = f.root.Indices.read()
    return indices


def writeback_target_indices(indices, targets):
    with hdf.open_file(targets, mode='r+') as f:
        if f.__contains__('/Indices'):
            # check my indices are the same?
            log.info("skipping writing taget indices... already exist")
        else:
            log.info("writing target indices back to target file")
            f.create_array("/", "Indices", obj=indices)


def points_to_hdf(lonlat, outfile, fieldname=None, fieldvals=None):

    with hdf.open_file(outfile, 'w') as f:
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

    with hdf.open_file(filename, mode='r') as f:
        vals = [v for v in f.root.field]

    return np.array(vals)


def construct_splits(npixels, nchunks, overlap=0):
    # Build the equivalent windowed image
    # y bounds are INCLUSIVE
    y_arrays = np.array_split(np.arange(npixels), nchunks)
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

    # the +1 because we want pixel corner 1 beyond the last pixel
    lon_range = T1[2] + np.array([0, raster.width + 1]) * T1[0]
    lat_range = T1[5] + np.array([0, raster.height + 1]) * T1[4]

    lon_range = np.sort(lon_range)
    lat_range = np.sort(lat_range)

    return lon_range, lat_range


class Image:
    def __init__(self, filename, chunk_idx=0, nchunks=1, overlap=0):
        assert chunk_idx >= 0 and chunk_idx < nchunks

        self.chunk_idx = chunk_idx
        self.nchunks = nchunks
        self.filename = filename

        # Get the full image details
        with rasterio.open(self.filename, 'r') as geotiff:
            # self._full_xrange, self._full_yrange = bounding_box(geotiff)
            self._full_res = (geotiff.width, geotiff.height, geotiff.count)
            self._nodata_value = geotiff.meta['nodata']
            # we don't support different channels with different dtypes
            for d in geotiff.dtypes[1:]:
                if geotiff.dtypes[0] != d:
                    raise ValueError("No support for multichannel geotiffs "
                                     "with differently typed channels")
            self._dtype = np.dtype(geotiff.dtypes[0])

            # Build the affine transformation for the FULL image
            A = geotiff.affine

        # No shearing or rotation allowed!!
        if not ((A[1] == 0) and (A[3] == 0)):
            raise RuntimeError("Transform to pixel coordinates"
                               "has rotation or shear")

        # TODO clean this up into a function
        self.pixsize_x = A[0]
        self.pixsize_y = A[4]
        self._y_flipped = self.pixsize_y < 0
        self._start_lon = A[2]
        self._start_lat = A[5]

        # construct the canonical pixel<->position map
        pix_x = range(self._full_res[0] + 1 + 1)  # 1 past corner of last pixel
        coords_x = [self._start_lon + float(k) * self.pixsize_x
                    for k in pix_x]
        self._coords_x = coords_x
        pix_y = range(self._full_res[1] + 1 + 1)  # ditto
        coords_y = [self._start_lat + float(k) * self.pixsize_y
                    for k in pix_y]
        self._coords_y = coords_y
        self._pix_x_to_coords = dict(zip(pix_x, coords_x))
        self._pix_y_to_coords = dict(zip(pix_y, coords_y))

        # inclusive y range of this chunk in full image
        ymin, ymax = construct_splits(self._full_res[1],
                                      nchunks, overlap)[chunk_idx]
        self._offset = np.array([0, ymin], dtype=int)
        # inclusive x range of this chunk (same for all chunks)
        xmin, xmax = 0, self._full_res[0] - 1

        assert(xmin < xmax)
        assert(ymin < ymax)

        # get resolution of this chunk
        xres = self._full_res[0]
        yres = ymax - ymin + 1  # note the +1 because inclusive bounds

        # Calculate the new values for resolution and bounding box
        self.resolution = (xres, yres, self._full_res[2])

        start_bound_x, start_bound_y = self._global_pix2lonlat(
            np.array([[xmin, ymin]]))[0]
        # one past the last pixel (note the +1)
        outer_bound_x, outer_bound_y = self._global_pix2lonlat(
            np.array([[xmax + 1, ymax + 1]]))[0]

        assert(start_bound_x < outer_bound_x)
        if self._y_flipped:
            assert(start_bound_y > outer_bound_y)
            self.bbox = [[start_bound_x, outer_bound_x],
                         [outer_bound_y, start_bound_y]]
        else:
            assert(start_bound_y < outer_bound_y)
            self.bbox = [[start_bound_x, outer_bound_x],
                         [start_bound_y, outer_bound_y]]

    def __repr__(self):
        return "<geo.Image({}), chunk {} of {})>".format(self.filename,
                                                         self.chunk_idx,
                                                         self.nchunks)

    def data(self):
        # ((ymin, ymax),(xmin, xmax))
        window = ((self._offset[1], self._offset[1] + self.resolution[1]),
                  (self._offset[0], self._offset[0] + self.resolution[0]))
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

    # @contract(xy='array[Nx2](int64),N>0')
    def _global_pix2lonlat(self, xy):
        result = np.array([[self._pix_x_to_coords[x],
                           self._pix_y_to_coords[y]] for x, y in xy])
        return result

    # @contract(xy='array[Nx2](int64),N>0')
    def pix2lonlat(self, xy):
        result = self._global_pix2lonlat(xy + self._offset)
        return result

    # @contract(lonlat='array[Nx2](float64),N>0')
    def _global_lonlat2pix(self, lonlat):
        x = np.searchsorted(self._coords_x, lonlat[:, 0], side='right') - 1
        x = x.astype(int)
        ycoords = self._coords_y[::-1] if self._y_flipped else self._coords_y
        side = 'left' if self._y_flipped else 'right'
        y = np.searchsorted(ycoords, lonlat[:, 1], side=side) - 1
        y = self._full_res[1] - y if self._y_flipped else y
        y = y.astype(int)

        # We want the *closed* interval, which means moving
        # points on the end back by 1
        on_end_x = lonlat[:, 0] == self._coords_x[-1]
        on_end_y = lonlat[:, 1] == self._coords_y[-1]
        x[on_end_x] -= 1
        y[on_end_y] -= 1

        assert all(np.logical_and(x >= 0, x < self._full_res[0]))
        assert all(np.logical_and(y >= 0, y < self._full_res[1]))

        result = np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)
        return result

    # @contract(lonlat='array[Nx2](float64),N>0')
    def lonlat2pix(self, lonlat):
        result = self._global_lonlat2pix(lonlat) - self._offset
        # check the postcondition
        x = result[:, 0]
        y = result[:, 1]
        assert all(np.logical_and(x >= 0, x < self.resolution[0]))
        assert all(np.logical_and(y >= 0, y < self.resolution[1]))
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

# def bbox2affine(xmax, xmin, ymax, ymin, xres, yres):

#     pixsize_x = (xmax - xmin) / (xres + 1)
#     pixsize_y = (ymax - ymin) / (yres + 1)

#     A = Affine(pixsize_x, 0, xmin,
#                0, -pixsize_y, ymax)

#     return A, pixsize_x, pixsize_y


def output_filename(feature_name, chunk_index, output_dir):
    filename = feature_name + ".part{}.hdf5".format(chunk_index)
    full_path = os.path.join(output_dir, filename)
    return full_path


def output_features(feature_vector, outfile, featname="features",
                    shape=None, bbox=None):
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
        shape: tuple, optional
            The original shape of the feature for reproducing an image
        bbox: ndarray, optional
            The bounding box of the original data for reproducing an image
    """
    h5file = hdf.open_file(outfile, mode='w')

    # Make sure we are writing "long" arrays
    if feature_vector.ndim < 2:
        feature_vector = feature_vector[:, np.newaxis]
    array_shape = feature_vector.shape

    filters = hdf.Filters(complevel=5, complib='zlib')

    if np.ma.isMaskedArray(feature_vector):
        fobj = feature_vector.data
        if np.ma.count_masked(feature_vector) == 0:
            fmask = np.zeros(array_shape, dtype=bool)
        else:
            fmask = feature_vector.mask
    else:
        fobj = feature_vector
        fmask = np.zeros(array_shape, dtype=bool)

    h5file.create_carray("/", featname, filters=filters,
                         atom=hdf.Float64Atom(), shape=array_shape, obj=fobj)
    h5file.create_carray("/", "mask", filters=filters,
                         atom=hdf.BoolAtom(), shape=array_shape, obj=fmask)

    if shape is not None:
        h5file.getNode('/' + featname).attrs.shape = shape
        h5file.root.mask.attrs.shape = shape
    if bbox is not None:
        h5file.getNode('/' + featname).attrs.bbox = bbox
        h5file.root.mask.attrs.bbox = bbox

    h5file.close()


def load_and_cat(hdf5_vectors):
    data_list = []
    for filename in hdf5_vectors:
        with hdf.open_file(filename, mode='r') as f:
            data = f.root.features[:]
            mask = f.root.mask[:]
            a = np.ma.masked_array(data=data, mask=mask)
            data_list.append(a)
    all_data = np.ma.concatenate(data_list, axis=1)
    return all_data


def load_attributes(filename_dict):
    # Only bother loading the first one as they're all the same for now
    fname = filename_dict[0][0]
    shape = None
    bbox = None
    with hdf.open_file(fname, mode='r') as f:
        if 'shape' in f.root.features.attrs:
            shape = f.root.features.attrs.shape
        if 'bbox' in f.root.features.attrs:
            bbox = f.root.features.attrs.bbox
    return shape, bbox
