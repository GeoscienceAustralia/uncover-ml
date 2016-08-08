from __future__ import division

import os.path
import logging
import time
import json
import pickle
from functools import partial
from abc import ABCMeta, abstractmethod

import rasterio
import numpy as np
import shapefile
import tables as hdf

from uncoverml import mpiops
from uncoverml import image
from uncoverml import datatypes


log = logging.getLogger(__name__)


def load_settings(settings_file):
    with open(settings_file, 'rb') as f:
        s = pickle.load(f)
    return s


def save_settings(s, settings_file):
    with open(settings_file, 'wb') as f:
        pickle.dump(s, f)


def file_indices_okay(filenames):

    # get just the name eg /path/to/file_0.hdf5 -> file_0
    basenames = [os.path.splitext(os.path.basename(k))[0] for k in filenames]

    # file.part0of3 -> [file,0]
    split_total = [k.rsplit('of', 1) for k in basenames]
    try:
        totals = set([int(k[1]) for k in split_total])
    except:
        log.error("Filename does not contain total number of parts.")
        return False

    if len(totals) > 1:
        log.error("Files disagree about total chunks")
        return False

    total = totals.pop()

    minus_total = [k[0] for k in split_total]
    base_and_idx = [k.rsplit('.part', 1) for k in minus_total]
    bases = set([k[0] for k in base_and_idx])
    log.info("Input file sets: {}".format(set(bases)))

    # check every base has the right indices
    # "[[file,0], [file,1]] -> {file:[0,1]}
    try:
        base_ids = {k: set([int(j[1])
                           for j in base_and_idx if j[0] == k]) for k in bases}
    except:
        log.error("One or more filenames are not in <name>.part<idx>of<total>"
                  ".hdf5 format")
        # either there are no ints at the end or no underscore
        return False

    # Ensure all files are present
    true_set = set(range(1, total + 1))
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

    split_total = [k.rsplit('of', 1) for k in basenames]
    minus_total = [k[0] for k in split_total]
    indices = [int(k.rsplit('.part', 1)[1]) - 1 for k in minus_total]

    d = {i: [] for i in set(indices)}
    for i, f in zip(indices, sorted_filenames):
        d[i].append(f)
    return d


def points_from_hdf(filename, fieldnames):
    """
    TODO
    """
    vals = {}
    with hdf.open_file(filename, mode='r') as f:
        for fld in fieldnames:
            vals[fld] = (f.get_node("/" + fld).read())

    return vals


def points_to_hdf(outfile, fielddict={}):

    with hdf.open_file(outfile, 'w') as f:
        for fld, v in fielddict.items():
            f.create_array("/", fld, obj=v)


def write_targets(targets, filename):
    # takes a Target class
    # Make field dict for writing to HDF5
    fielddict = {
        'targets': targets._observations_unsorted,
        'Longitude': targets._positions_unsorted[:, 0],
        'Latitude': targets._positions_unsorted[:, 1],
        'FoldIndices': targets._folds_unsorted,
        'Targets_sorted': targets.observations,
        'Positions_sorted': targets.positions,
        'FoldIndices_sorted': targets.folds
    }
    points_to_hdf(filename, fielddict)


def load_targets(filename):
    fields = ['Targets_sorted', 'Positions_sorted', 'FoldIndices_sorted']
    fielddict = points_from_hdf(filename, fields)
    positions = fielddict['Positions_sorted']
    observations = fielddict['Targets_sorted']
    folds = fielddict['FoldIndices_sorted']
    result = datatypes.CrossValTargets(positions, observations, folds)
    return result


class ImageSource(metaclass=ABCMeta):

    @abstractmethod
    def data(self, min_x, max_x, min_y, max_y):
        pass

    @property
    def full_resolution(self):
        return self._full_res

    @property
    def dtype(self):
        return self._dtype

    @property
    def nodata_value(self):
        return self._nodata_value

    @property
    def pixsize_x(self):
        return self._pixsize_x

    @property
    def pixsize_y(self):
        return self._pixsize_y

    @property
    def origin_latitude(self):
        return self._start_lat

    @property
    def origin_longitude(self):
        return self._start_lon


class RasterioImageSource(ImageSource):

    def __init__(self, filename):

        self._filename = filename
        with rasterio.open(self._filename, 'r') as geotiff:
            self._full_res = (geotiff.width, geotiff.height, geotiff.count)
            self._nodata_value = geotiff.meta['nodata']
            # we don't support different channels with different dtypes
            for d in geotiff.dtypes[1:]:
                if geotiff.dtypes[0] != d:
                    raise ValueError("No support for multichannel geotiffs "
                                     "with differently typed channels")
            self._dtype = np.dtype(geotiff.dtypes[0])

            A = geotiff.affine
            # No shearing or rotation allowed!!
            if not ((A[1] == 0) and (A[3] == 0)):
                raise RuntimeError("Transform to pixel coordinates"
                                   "has rotation or shear")
            self._pixsize_x = A[0]
            self._pixsize_y = A[4]
            self._start_lon = A[2]
            self._start_lat = A[5]

            self._y_flipped = self._pixsize_y < 0
            if self._y_flipped:
                self._start_lat += self._pixsize_y * self._full_res[1]
                self._pixsize_y *= -1

    def data(self, min_x, max_x, min_y, max_y):

        if self._y_flipped:
            min_y_new = self._full_res[1] - max_y
            max_y_new = self._full_res[1] - min_y
            min_y = min_y_new
            max_y = max_y_new

        # NOTE these are exclusive
        window = ((min_y, max_y), (min_x, max_x))
        with rasterio.open(self._filename, 'r') as geotiff:
            d = geotiff.read(window=window, masked=True)
        d = d[np.newaxis, :, :] if d.ndim == 2 else d
        d = np.ma.transpose(d, [2, 1, 0])  # Transpose and channels at back

        if self._y_flipped:
            d = d[:, ::-1]

        # Otherwise scikit image complains
        m = np.ma.MaskedArray(data=np.ascontiguousarray(d.data),
                              mask=np.ascontiguousarray(d.mask))

        # uniform mask format
        if np.ma.count_masked(m) == 0:
            m = np.ma.masked_array(data=m.data,
                                   mask=np.zeros_like(m.data, dtype=bool))
        assert m.data.ndim == 3
        assert m.mask.ndim == 3
        return m


class ArrayImageSource(ImageSource):
    """
    An image source that uses an internally stored numpy array

    Parameters
    ----------
    A : MaskedArray
        masked array of shape (xpix, ypix, channels) that contains the
        image data.
    origin : ndarray
        Array of the form [lonmin, latmin] that defines the origin of the image
    pixsize : ndarray
        Array of the form [pixsize_x, pixsize_y] defining the size of a pixel
    """
    def __init__(self, A, origin, pixsize):
        self._data = A
        self._full_res = A.shape
        self._dtype = A.dtype
        self._nodata_value = A.fill_value
        self._pixsize_x = pixsize[0]
        self._pixsize_y = pixsize[1]
        self._start_lon = origin[0]
        self._start_lat = origin[1]

    def data(self, min_x, max_x, min_y, max_y):
        # MUST BE EXCLUSIVE
        data_window = self._data[min_x:max_x, :][:, min_y:max_y]
        return data_window


def output_filename(feature_name, chunk_index, n_chunks, output_dir):
    filename = feature_name + ".part{}of{}.hdf5".format(chunk_index + 1,
                                                        n_chunks)
    full_path = os.path.join(output_dir, filename)
    return full_path


def output_blank(filename, shape=None, bbox=None):
    with hdf.open_file(filename, mode='w') as h5file:
        h5file.root._v_attrs["blank"] = True
        if shape is not None:
            h5file.root._v_attrs["image_shape"] = shape
        if bbox is not None:
            h5file.root._v_attrs["image_bbox"] = bbox


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
    if feature_vector is None:
        output_blank(outfile, shape, bbox)
        return

    log.info("writing {} array with name {}".format(
        feature_vector.shape, outfile))
    with hdf.open_file(outfile, mode='w') as h5file:
        h5file.root._v_attrs["blank"] = False

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
                             atom=hdf.Float64Atom(), shape=array_shape,
                             obj=fobj)

        h5file.create_carray("/", "mask", filters=filters,
                             atom=hdf.BoolAtom(), shape=array_shape, obj=fmask)

        if shape is not None:
            h5file.root._v_attrs["image_shape"] = shape
        if bbox is not None:
            h5file.root._v_attrs["image_bbox"] = bbox

    start = time.time()
    file_exists = False

    while not file_exists and (time.time() - start) < 5:

        file_exists = os.path.exists(outfile)
        time.sleep(0.1)

    if not file_exists:
        raise RuntimeError("{} never written!".format(outfile))

    return True


def load_and_cat(hdf5_vectors):
    data_shapes = []
    # pass one to get the shapes
    for filename in hdf5_vectors:
        with hdf.open_file(filename, mode='r') as f:
            if f.root._v_attrs["blank"]:  # no data in this chunk
                return None
            data_shapes.append(f.root.features.shape)

    # allocate memory
    x_shps, y_shps = zip(*data_shapes)
    x_shp = set(x_shps).pop()
    y_shp = np.sum(np.array(y_shps))

    log.info("Allocating shape {}, mem {}".format((x_shp, y_shp),
                                                  x_shp * y_shp * 72. / 1e9))

    all_data = np.empty((x_shp, y_shp), dtype=float)
    all_mask = np.empty((x_shp, y_shp), dtype=bool)

    # read files in
    start_idx = 0
    end_idx = -1
    for filename in hdf5_vectors:
        with hdf.open_file(filename, mode='r') as f:
            end_idx = start_idx + f.root.features.shape[1]
            all_data[:, start_idx:end_idx] = f.root.features[:]
            all_mask[:, start_idx:end_idx] = f.root.mask[:]
            start_idx = end_idx

    result = np.ma.masked_array(data=all_data, mask=all_mask)
    return result


def load_attributes(filename_dict):
    # Only bother loading the first one as they're all the same for now
    fname = filename_dict[0][0]
    shape = None
    bbox = None
    with hdf.open_file(fname, mode='r') as f:
        if 'image_shape' in f.root._v_attrs:
            shape = f.root._v_attrs.image_shape
        if 'image_bbox' in f.root._v_attrs:
            bbox = f.root._v_attrs.image_bbox
    return shape, bbox


def load_shapefile(filename, field):
    """
    TODO
    """

    sf = shapefile.Reader(filename)
    fdict = {f[0]: i for i, f in enumerate(sf.fields[1:])}  # Skip DeletionFlag

    if field not in fdict:
        raise ValueError("Requested field is not in records!")

    vind = fdict[field]
    vals = np.array([r[vind] for r in sf.records()])
    coords = []
    for shape in sf.iterShapes():
        coords.append(list(shape.__geo_interface__['coordinates']))
    label_coords = np.array(coords)
    return label_coords, vals


class ImageWriter:
    def __init__(self, shape, bbox, name, n_subchunks, outputdir):
        # affine
        self.A, _, _ = image.bbox2affine(bbox[1, 0], bbox[0, 0],
                                         bbox[0, 1], bbox[1, 1],
                                         shape[0], shape[1])
        self.shape = shape
        self.bbox = bbox
        self.name = name
        self.outputdir = outputdir
        self.output_filename = os.path.join(outputdir, name + ".tif")
        self.n_subchunks = n_subchunks
        log.info("Imwriter image shape: {}".format(self.shape))
        log.info("Imwriter number of subchunks: {}".format(mpiops.chunks
                                               * self.n_subchunks))
        self.sub_starts = [k[0] for k in np.array_split(
                           np.arange(self.shape[1]),
                           mpiops.chunks * self.n_subchunks)]
        if mpiops.chunk_index == 0:
            self.f = rasterio.open(self.output_filename, 'w', driver='GTiff',
                                   width=self.shape[0], height=self.shape[1],
                                   dtype=np.float64, count=self.shape[2],
                                   transform=self.A)

    def write(self, x, subchunk_index):
        rows = self.shape[0]
        bands = x.shape[1]
        image = x.reshape((rows, -1, bands))

        mpiops.comm.barrier()
        log.info("writing commenced!")
        if mpiops.chunk_index != 0:
            log.info("node {} sending..".format(mpiops.chunk_index))
            mpiops.comm.send(image, dest=0)
            log.info("node {} sent!".format(mpiops.chunk_index))
        else:
            for node in range(mpiops.chunks):
                node = mpiops.chunks - node - 1
                subindex = node * self.n_subchunks + subchunk_index
                ystart = self.sub_starts[subindex]
                log.info("node 0 receiveing from {}".format(node))
                data = mpiops.comm.recv(source=node) \
                    if node != 0 else image
                log.info("node 0 received from {}!".format(node))
                data = np.ma.transpose(data, [2, 1, 0])  # untranspose
                yend = ystart + data.shape[1]  # this is Y
                window = ((ystart, yend), (0, self.shape[0]))
                index_list = list(range(1, bands + 1))
                log.info("write data index_list: {}".format(index_list))
                self.f.write(data, window=window, indexes=index_list)
        mpiops.comm.barrier()


def export_scores(scores, y, Ey, filename):

    log.info("{} Scores".format(filename.rstrip(".json")))
    for metric, score in scores.items():
        log.info("{} = {}".format(metric, score))

    with open(filename, 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)
