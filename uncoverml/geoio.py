from __future__ import division

import os.path
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import json
import pickle
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject
from affine import Affine
import numpy as np
import shapefile
import tables as hdf

from uncoverml import mpiops
from uncoverml import image
from uncoverml import features
from uncoverml.transforms import missing_percentage
from uncoverml.targets import Targets


log = logging.getLogger(__name__)


_lower_is_better = ['mll', 'msll', 'smse', 'log_loss']


class ImageSource:
    __metaclass__ = ABCMeta

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

    @property
    def crs(self):
        return self._crs


class RasterioImageSource(ImageSource):

    def __init__(self, filename):

        self._filename = filename
        assert os.path.isfile(filename), '{} does not exist'.format(filename)
        with rasterio.open(self._filename, 'r') as geotiff:
            self._full_res = (geotiff.width, geotiff.height, geotiff.count)
            self._nodata_value = geotiff.meta['nodata']
            # we don't support different channels with different dtypes
            for d in geotiff.dtypes[1:]:
                if geotiff.dtypes[0] != d:
                    raise ValueError("No support for multichannel geotiffs "
                                     "with differently typed channels")
            self._dtype = np.dtype(geotiff.dtypes[0])
            self._crs = geotiff.crs

            A = geotiff.transform
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

        # if nans exist in data, mask them, i.e. convert to nodatavalue
        # TODO: Consider removal once covariates are fixed
        nans = np.isnan(d.data)
        if d.mask.ndim:
            d.mask[nans] = True

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
    def __init__(self, A, origin, crs, pixsize):
        self._data = A
        self._full_res = A.shape
        self._dtype = A.dtype
        self._nodata_value = A.fill_value
        self._pixsize_x = pixsize[0]
        self._pixsize_y = pixsize[1]
        self._start_lon = origin[0]
        self._start_lat = origin[1]
        self._crs = crs

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
    dtypes = ['float' if (k[0] == 'N' or k[0] == 'F') else '<U{}'.format(k[1])
              for k in dtype_flags]
    records = np.array(sf.records()).T
    record_dict = {k: np.array(r, dtype=d) for k, r, d in zip(
        shapefields, records, dtypes)}
    if targetfield in record_dict:
        val = record_dict.pop(targetfield)
    else:
        raise ValueError("Can't find target property in shapefile." +
                         "Candidates: {}".format(record_dict.keys()))
    othervals = record_dict

    # Get coordinates
    coords = []
    for shape in sf.iterShapes():
        coords.append(list(shape.__geo_interface__['coordinates']))
    label_coords = np.array(coords).squeeze()
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
    crs = template_image.crs
    return eff_shape, eff_bbox, crs


class ImageWriter:

    nodata_value = np.array(-1e20, dtype='float32')

    def __init__(self, shape, bbox, crs, name, n_subchunks, outputdir,
                 band_tags=None, independent=False):
        # affine
        self.A, _, _ = image.bbox2affine(bbox[1, 0], bbox[0, 0],
                                         bbox[0, 1], bbox[1, 1],
                                         shape[0], shape[1])
        self.shape = shape
        self.outbands = len(band_tags)
        self.bbox = bbox
        self.name = name
        self.outputdir = outputdir
        self.n_subchunks = n_subchunks
        self.independent = independent  # mpi control
        self.sub_starts = [k[0] for k in np.array_split(
                           np.arange(self.shape[1]),
                           mpiops.chunks * self.n_subchunks)]

        # file tags don't have spaces
        if band_tags:
            file_tags = ["_".join(k.lower().split()) for k in band_tags]
        else:
            file_tags = [str(k) for k in range(self.outbands)]
            band_tags = file_tags

        files = []
        file_names = []

        for band in range(self.outbands):
            output_filename = os.path.join(outputdir, name + "_" +
                                           file_tags[band] + ".tif")
            f = rasterio.open(output_filename, 'w', driver='GTiff',
                              width=self.shape[0], height=self.shape[1],
                              dtype=np.float32, count=1,
                              crs=crs,
                              transform=self.A,
                              nodata=self.nodata_value,
                              compress='lzw',
                              bigtiff='YES',
                              tiled=True
                              )
            f.update_tags(1, image_type=band_tags[band])
            files.append(f)
            file_names.append(output_filename)

        if independent:
            self.files = files
        else:
            if mpiops.chunk_index == 0:
                # create a file for each band
                self.files = files
                self.file_names = file_names
            else:
                self.file_names = []

            self.file_names = mpiops.comm.bcast(self.file_names, root=0)

    def write(self, x, subchunk_index):
        """
        :param x:
        :param subchunk_index:
        :param independent: bool
            independent image writing by different processes, i.e., images are not chunked
        :return:
        """
        x = x.astype(np.float32)
        rows = self.shape[0]
        bands = x.shape[1]

        image = x.reshape((rows, -1, bands))
        # make sure we're writing nodatavals
        if x.mask is not False:
            x.data[x.mask] = self.nodata_value

        mpiops.comm.barrier()
        log.info("Writing partition to output file")

        if self.independent:
            data = np.ma.transpose(image, [2, 1, 0])  # untranspose
            # write each band separately
            for i, f in enumerate(self.files):
                f.write(data[i:i+1], compress='lzw',
                        bigtiff='YES')
        else:
            if mpiops.chunk_index != 0:
                mpiops.comm.send(image, dest=0)
            else:
                for node in range(mpiops.chunks):
                    node = mpiops.chunks - node - 1
                    subindex = mpiops.chunks*subchunk_index + node
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

    def close(self):  # we can explicitly close rasters using this
        if mpiops.chunk_index == 0:
            for f in self.files:
                f.close()
        mpiops.comm.barrier()

    def output_thumbnails(self, ratio=10):
        this_chunk_files = np.array_split(self.file_names,
                                          mpiops.chunks)[mpiops.chunk_index]
        for f in this_chunk_files:
            thumbnails = os.path.splitext(f)
            thumbnail = thumbnails[0] + '_thumbnail' + thumbnails[1]
            resample(f, output_tif=thumbnail, ratio=ratio)


def feature_names(config):

    results = []
    for s in config.feature_sets:
        feats = []
        for tif in s.files:
            name = os.path.basename(tif)
            feats.append(name)
        feats.sort()
        results += feats
    return results


def _iterate_sources(f, config):

    results = []
    for s in config.feature_sets:
        extracted_chunks = {}
        for tif in s.files:
            name = os.path.abspath(tif)
            image_source = RasterioImageSource(tif)
            x = f(image_source)
            # TODO this may hurt performance. Consider removal
            if type(x) is np.ma.MaskedArray:
                count = mpiops.count(x)
                # if not np.all(count > 0):
                #     s = ("{} has no data in at least one band.".format(name) +
                #          " Valid_pixel_count: {}".format(count))
                #     raise ValueError(s)
                missing_percent = missing_percentage(x)
                t_missing = mpiops.comm.allreduce(
                    missing_percent) / mpiops.chunks
                log.info("{}: {}px {:2.2f}% missing".format(
                    name, count, t_missing))
            extracted_chunks[name] = x
        extracted_chunks = OrderedDict(sorted(
            extracted_chunks.items(), key=lambda t: t[0]))

        results.append(extracted_chunks)
    return results


def image_resolutions(config):
    def f(image_source):
        r = image_source._full_res
        return r

    result = _iterate_sources(f, config)
    return result


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

    frac = config.subsample_fraction

    def f(image_source):
        r_t = features.extract_features(image_source, targets, n_subchunks=1,
                                        patchsize=config.patchsize)
        r_a = features.extract_subchunks(image_source, subchunk_index=0,
                                         n_subchunks=1,
                                         patchsize=config.patchsize)
        if frac < 1.0:
            np.random.seed(1)
            r_a = r_a[np.random.rand(r_a.shape[0]) < frac]

        r_data = np.concatenate([r_t.data, r_a.data], axis=0)
        r_mask = np.concatenate([r_t.mask, r_a.mask], axis=0)
        r = np.ma.masked_array(data=r_data, mask=r_mask)
        return r
    result = _iterate_sources(f, config)
    return result


def unsupervised_feature_sets(config):

    frac = config.subsample_fraction

    def f(image_source):
        r = features.extract_subchunks(image_source, subchunk_index=0,
                                       n_subchunks=1,
                                       patchsize=config.patchsize)
        if frac < 1.0:
            np.random.seed(1)
            r = r[np.random.rand(r.shape[0]) < frac]
        return r
    result = _iterate_sources(f, config)
    return result


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

        # plot the results
        plt.figure()
        plt.plot(range(len(sorted_features)), sorted_scores)
        plt.xticks(range(len(sorted_features)), sorted_features,
                   rotation='vertical')
        plt.savefig('{}.png'.format(measure))

    # Write the results out to a file
    with open(outfile_ranks, 'w') as output_file:
        json.dump(score_listing, output_file, sort_keys=True, indent=4)


def export_model(model, config):
    outfile_state = os.path.join(config.output_dir,
                                 config.name + ".model")
    # TODO: investigate why catboost model does not save target transform
    state_dict = {"model": model,
                  "config": config}
    # if config.algorithm == 'catboost':
    #     state_dict['target_transform'] = model.target_transform

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

    # Make sure we convert numpy arrays to lists
    scores = {s: v if np.isscalar(v) else v.tolist()
              for s, v in crossval_output.scores.items()}

    with open(outfile_scores, 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)

    outfile_results = os.path.join(config.output_dir,
                                   config.name + "_results.hdf5")
    with hdf.open_file(outfile_results, 'w') as f:
        for fld, v in crossval_output.y_pred.items():
            label = _make_valid_array_name(fld)
            f.create_array("/", label, obj=v.data)
            f.create_array("/", label + "_mask", obj=v.mask)
        f.create_array("/", "y_true", obj=crossval_output.y_true)

    if not crossval_output.classification:
        create_scatter_plot(outfile_results, config)


def _make_valid_array_name(label):
    label = "_".join(label.split())
    label = ''.join(filter(str.isalnum, label))  # alphanum only
    if label[0].isdigit():
        label = '_' + label
    return label


def create_scatter_plot(outfile_results, config):
    true_vs_pred = os.path.join(config.output_dir,
                                config.name + "_results.csv")
    true_vs_pred_plot = os.path.join(config.output_dir,
                                     config.name + "_results.png")
    with hdf.open_file(outfile_results, 'r') as f:
        prediction = f.get_node("/", "Prediction").read()
        y_true = f.get_node("/", "y_true").read()
        np.savetxt(true_vs_pred, X=np.vstack([y_true, prediction]).T,
                   delimiter=',')
        plt.figure()
        plt.scatter(y_true, prediction)
        plt.title('true vs prediction')
        plt.xlabel('True')
        plt.ylabel('Prediction')
        plt.savefig(true_vs_pred_plot)


def resample(input_tif, output_tif, ratio, resampling=5):
    """
    Parameters
    ----------
    input_tif: str or rasterio.io.DatasetReader
        input file path or rasterio.io.DatasetReader object
    output_tif: str
        output file path
    ratio: float
        ratio by which to shrink/expand
        ratio > 1 means shrink
    resampling: int, optional
        default is 5 (average) resampling. Other options are as follows:
        nearest = 0
        bilinear = 1
        cubic = 2
        cubic_spline = 3
        lanczos = 4
        average = 5
        mode = 6
        gauss = 7
        max = 8
        min = 9
        med = 10
        q1 = 11
        q3 = 12
    """

    src = rasterio.open(input_tif, mode='r')

    nodatavals = src.get_nodatavals()
    new_shape = round(src.height / ratio), round(src.width / ratio)
    # adjust the new affine transform to the smaller cell size
    aff = src.get_transform()

    # c, a, b, f, d, e, works on rasterio versions >=1.0
    # newaff = Affine(aff.a * ratio, aff.b, aff.c,
    #                 aff.d, aff.e * ratio, aff.f)
    #
    newaff = Affine(aff[0] * ratio, aff[1], aff[2],
                    aff[3], aff[4] * ratio, aff[5])

    dest = rasterio.open(output_tif, 'w', driver='GTiff',
                         height=new_shape[0], width=new_shape[1],
                         count=src.count, dtype=rasterio.float32,
                         crs=src.crs, transform=newaff,
                         nodata=nodatavals[0])
    for b in range(src.count):
        arr = src.read(b+1)
        new_arr = np.empty(shape=new_shape, dtype=arr.dtype)
        reproject(arr, new_arr,
                  src_transform=aff,
                  dst_transform=newaff,
                  src_crs=src.crs,
                  src_nodata=nodatavals[b],
                  dst_crs=src.crs,
                  dst_nodata=nodatavals[b],
                  resample=resampling)
        dest.write(new_arr, b + 1)
    src.close()
    dest.close()
