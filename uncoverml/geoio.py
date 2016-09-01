from __future__ import division

import os.path
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import json
import pickle

import rasterio
import numpy as np
import shapefile
import tables as hdf

from uncoverml import mpiops
from uncoverml import image
from uncoverml import features
from uncoverml.targets import Targets


log = logging.getLogger(__name__)


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

        # # uniform mask format
        # if np.ma.count_masked(m) == 0:
        #     m = np.ma.masked_array(data=m.data,
        #                            mask=np.zeros_like(m.data, dtype=bool))
        assert m.data.ndim == 3
        assert m.mask.ndim == 3 or m.mask.ndim == 0
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


def load_shapefile(filename, targetfield):
    """
    TODO
    """
    sf = shapefile.Reader(filename)
    shapefields = [f[0] for f in sf.fields[1:]]  # Skip DeletionFlag
    dtype_flags = [(f[1], f[2]) for f in sf.fields[1:]]  # Skip DeletionFlag
    dtypes = ['float' if k[0] == 'N' else '<U{}'.format(k[1])
              for k in dtype_flags]
    records = np.array(sf.records()).T
    record_dict = {k: np.array(r, dtype=d) for k, r, d in zip(
        shapefields, records, dtypes)}
    val = record_dict.pop(targetfield)
    othervals = record_dict

    # Get coordinates
    coords = []
    for shape in sf.iterShapes():
        coords.append(list(shape.__geo_interface__['coordinates']))
    label_coords = np.array(coords)

    return label_coords, val, othervals


def load_targets(shapefile, targetfield):
    """
    Loads the shapefile onto node 0 then distributes it across all
    available nodes
    """
    if mpiops.chunk_index == 0:
        lonlat, vals, othervals = load_shapefile(shapefile, targetfield)
        # sort by y then x
        ordind = np.lexsort(lonlat.T)
        vals = vals[ordind]
        lonlat = lonlat[ordind]
        for k, v in othervals.items():
            othervals[k] = v[ordind]

        lonlat = np.array_split(lonlat, mpiops.chunks)
        vals = np.array_split(vals, mpiops.chunks)
        split_othervals = {k: np.array_split(v, mpiops.chunks)
                           for k, v in othervals.items()}
        othervals = [{k: v[i] for k, v in split_othervals.items()}
                     for i in range(mpiops.chunks)]
    else:
        lonlat, vals, othervals = None, None, None

    lonlat = mpiops.comm.scatter(lonlat, root=0)
    vals = mpiops.comm.scatter(vals, root=0)
    othervals = mpiops.comm.scatter(othervals, root=0)
    log.info("Node {} has been assigned {} targets".format(mpiops.chunk_index,
                                                           lonlat.shape[0]))
    targets = Targets(lonlat, vals, othervals=othervals)
    return targets


def get_image_spec(model, config):
    # temp workaround, we should have an image spec to check against
    nchannels = len(model.get_predict_tags())
    imagelike = config.feature_sets[0].files[0]
    template_image = image.Image(RasterioImageSource(imagelike))
    eff_shape = template_image.patched_shape(config.patchsize) + (nchannels,)
    eff_bbox = template_image.patched_bbox(config.patchsize)
    return eff_shape, eff_bbox


class ImageWriter:

    nodata_value = -1e23

    def __init__(self, shape, bbox, name, n_subchunks, outputdir,
                 band_tags=None):
        # affine
        self.A, _, _ = image.bbox2affine(bbox[1, 0], bbox[0, 0],
                                         bbox[0, 1], bbox[1, 1],
                                         shape[0], shape[1])
        self.shape = shape
        self.bbox = bbox
        self.name = name
        self.outputdir = outputdir
        self.n_subchunks = n_subchunks
        self.sub_starts = [k[0] for k in np.array_split(
                           np.arange(self.shape[1]),
                           mpiops.chunks * self.n_subchunks)]

        # file tags don't have spaces
        if band_tags:
            file_tags = ["_".join(k.lower().split()) for k in band_tags]
        else:
            file_tags = [str(k) for k in range(shape[2])]
            band_tags = file_tags

        if mpiops.chunk_index == 0:
            # create a file for each band
            self.files = []
            for band in range(self.shape[2]):
                output_filename = os.path.join(outputdir, name + "_" +
                                               file_tags[band] + ".tif")
                f = rasterio.open(output_filename, 'w', driver='GTiff',
                                  width=self.shape[0], height=self.shape[1],
                                  dtype=np.float64, count=1,
                                  transform=self.A,
                                  nodata=self.nodata_value)
                f.update_tags(1, image_type=band_tags[band])
                self.files.append(f)

    def write(self, x, subchunk_index):
        rows = self.shape[0]
        bands = x.shape[1]
        image = x.reshape((rows, -1, bands))
        # make sure we're writing nodatavals
        if x.mask is not False:
            x.data[x.mask] = self.nodata_value

        mpiops.comm.barrier()
        log.info("Writing partition to output file")
        if mpiops.chunk_index != 0:
            mpiops.comm.send(image, dest=0)
        else:
            for node in range(mpiops.chunks):
                node = mpiops.chunks - node - 1
                subindex = node * self.n_subchunks + subchunk_index
                ystart = self.sub_starts[subindex]
                data = mpiops.comm.recv(source=node) \
                    if node != 0 else image
                data = np.ma.transpose(data, [2, 1, 0])  # untranspose
                yend = ystart + data.shape[1]  # this is Y
                window = ((ystart, yend), (0, self.shape[0]))
                # write each band separately
                for i, f in enumerate(self.files):
                    f.write(data[i:i+1], window=window)
        mpiops.comm.barrier()


def _iterate_sources(f, config):

    results = []
    for s in config.feature_sets:
        extracted_chunks = {}
        for tif in s.files:
            name = os.path.basename(tif)
            log.info("Processing {}.".format(name))
            image_source = RasterioImageSource(tif)
            x = f(image_source)
            extracted_chunks[name] = x
        extracted_chunks = OrderedDict(sorted(
            extracted_chunks.items(), key=lambda t: t[0]))

        results.append(extracted_chunks)
    return results


def image_subchunks(subchunk_index, config):

    def f(image_source):
        r = features.extract_subchunks(image_source, subchunk_index,
                                       config.n_subchunks, config.patchsize)
        return r
    result = _iterate_sources(f, config)
    return result


def image_feature_sets(targets, config):

    def f(image_source):
        r = features.extract_features(image_source, targets,
                                      config.n_subchunks, config.patchsize)
        return r
    result = _iterate_sources(f, config)
    return result


def semisupervised_feature_sets(targets, config):

    def f(image_source):
        r_t = features.extract_features(image_source, targets, n_subchunks=1,
                                        patchsize=config.patchsize)
        r_a = features.extract_subchunks(image_source, subchunk_index=0,
                                         n_subchunks=1,
                                         patchsize=config.patchsize)
        r = np.ma.concatenate([r_t, r_a], axis=0)
        return r
    result = _iterate_sources(f, config)
    return result


def unsupervised_feature_sets(config):

    def f(image_source):
        r = features.extract_subchunks(image_source, subchunk_index=0,
                                       n_subchunks=1,
                                       patchsize=config.patchsize)
        return r
    result = _iterate_sources(f, config)
    return result

_lower_is_better = ['mll', 'msll', 'smse']


def export_feature_ranks(measures, feats, scores, config):
    outfile_ranks = os.path.join(config.output_dir,
                                 config.name + "_" + config.algorithm +
                                 "_featureranks.json")

    score_listing = dict(scores={}, ranks={})
    for measure, measure_scores in zip(measures, scores):

        # Sort the scores
        scores = sorted(zip(feats, measure_scores),
                        key=lambda s: s[1])
        if measure in _lower_is_better:
            scores.reverse()
        sorted_features, sorted_scores = zip(*scores)

        # Store the results
        score_listing['scores'][measure] = sorted_scores
        score_listing['ranks'][measure] = sorted_features

    # Write the results out to a file
    with open(outfile_ranks, 'w') as output_file:
        json.dump(score_listing, output_file, sort_keys=True, indent=4)


def export_model(model, config):
    outfile_state = os.path.join(config.output_dir,
                                 config.name + ".model")
    state_dict = {"model": model,
                  "config": config}
    with open(outfile_state, 'wb') as f:
        pickle.dump(state_dict, f)


def export_cluster_model(model, config):
    outfile_state = os.path.join(config.output_dir,
                                 config.name + ".cluster")
    state_dict = {"model": model,
                  "config": config}
    with open(outfile_state, 'wb') as f:
        pickle.dump(state_dict, f)


def export_crossval(crossval_output, config):
    outfile_scores = os.path.join(config.output_dir,
                                  config.name + "_scores.json")
    with open(outfile_scores, 'w') as f:
        json.dump(crossval_output.scores, f, sort_keys=True, indent=4)

    outfile_results = os.path.join(config.output_dir,
                                   config.name + "_results.hdf5")
    with hdf.open_file(outfile_results, 'w') as f:
        for fld, v in crossval_output.y_pred.items():
            label = "_".join(fld.split())
            f.create_array("/", label, obj=v.data)
            f.create_array("/", label + "_mask", obj=v.mask)
        f.create_array("/", "y_true", obj=crossval_output.y_true)
