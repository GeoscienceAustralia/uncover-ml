from __future__ import division
from typing import Optional
import joblib
import os.path
from subprocess import run
from pathlib import Path
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import cycle, islice
import json
from typing import Union
import pickle
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject
from rasterio.windows import Window
from xgboost import XGBRegressor
from sklearn.cluster import DBSCAN
from affine import Affine
import numpy as np
import shapefile
import tables as hdf
import pandas as pd
import geopandas as gpd

from uncoverml import mpiops
from uncoverml import image
from uncoverml import features
from uncoverml.config import Config
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

        self.filename = filename
        assert os.path.isfile(filename), '{} does not exist'.format(filename)

        with rasterio.open(self.filename, 'r') as geotiff:

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
        with rasterio.open(self.filename, 'r') as geotiff:
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


def load_shapefile(filename: str, targetfield: str):
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


def add_groups(lonlat, grouping_data, conf: Config):
    if grouping_data is not None:
        log.info(f"Grouping targets using {conf.group_col}")
        unique_groups = np.unique(grouping_data)  # np.unique returns sorted
        groups = np.zeros_like(grouping_data, dtype=np.uint16)
        for i, g in enumerate(unique_groups):
            groups[grouping_data == g] = i
        log.info(f"Found {max(groups) + 1} groups")
    else:
        log.info(f"No grouping column was supplied in config file.")
        log.info("Segmenting targets using DBSCAN clustering algorithm")
        dbscan = DBSCAN(eps=conf.groups_eps, n_jobs=-1, min_samples=10)

        dbscan.fit(lonlat)
        log.info("Finished segmentation!")
        groups = dbscan.labels_.astype(np.int16)
        log.info(f"Found {max(groups) + 1} groups + potential outliers in group -1")
    rc_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # list of colours
    colors = np.array(list(islice(cycle(rc_colors), int(max(groups) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    plt.figure(figsize=(16, 10))
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.scatter(lonlat[:, 0], lonlat[:, 1], s=10, c=colors[groups], cmap=colors)
    fig_file = conf.target_groups_file
    plt.savefig(fig_file)
    log.info(f"Saved groups in {fig_file}")
    return groups


def load_targets(shapefile, targetfield, conf: Config):
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

        if conf.group_targets:
            if conf.group_col:
                grouping_data = othervals[conf.group_col]
            else:
                grouping_data = None
            groups = add_groups(lonlat, grouping_data, conf)
        else:
            groups = np.ones_like(vals)

        if conf.weighted_model:
            weights = othervals[conf.weight_col_name]
        else:
            weights = np.ones_like(vals)

        lonlat = np.array_split(lonlat, mpiops.chunks)
        groups = np.array_split(groups, mpiops.chunks)
        vals = np.array_split(vals, mpiops.chunks)
        weights = np.array_split(weights, mpiops.chunks)
        split_othervals = {k: np.array_split(v, mpiops.chunks)
                           for k, v in othervals.items()}
        othervals = [{k: v[i] for k, v in split_othervals.items()}
                     for i in range(mpiops.chunks)]
    else:
        lonlat, vals, groups, weights, othervals = None, None, None, None, None

    lonlat = mpiops.comm.scatter(lonlat, root=0)
    groups = mpiops.comm.scatter(groups, root=0)
    vals = mpiops.comm.scatter(vals, root=0)
    weights = mpiops.comm.scatter(weights, root=0)
    othervals = mpiops.comm.scatter(othervals, root=0)
    log.info("Node {} has been assigned {} targets".format(mpiops.chunk_index,
                                                           lonlat.shape[0]))
    targets = Targets(lonlat, vals, groups, weights, othervals=othervals)
    return targets


def get_image_spec(model, config: Config):
    # temp workaround, we should have an image spec to check against
    nchannels = len(model.get_predict_tags())
    return get_image_spec_from_nchannels(nchannels, config)


def get_image_spec_from_nchannels(nchannels, config: Config):
    if config.prediction_template and config.is_prediction:
        imagelike = Path(config.prediction_template).absolute()
    else:
        imagelike = config.feature_sets[0].files[0]
    template_image = image.Image(RasterioImageSource(imagelike))
    eff_shape = template_image.patched_shape(config.patchsize) + (nchannels,)
    eff_bbox = template_image.patched_bbox(config.patchsize)
    crs = template_image.crs
    return eff_shape, eff_bbox, crs


class ImageWriter:

    nodata_value = np.array(-1e20, dtype='float32')

    def __init__(self, shape, bbox, crs, name, n_subchunks, outputdir,
                 band_tags=None, independent=False, **kwargs):
        """
        pass in additional geotif write options in kwargs
        """
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
        self.sub_starts = [k[0] for k in np.array_split(np.arange(self.shape[1]), mpiops.chunks * self.n_subchunks)]

        self.sub_ends = [k[-1] + 1 for k in np.array_split(np.arange(self.shape[1]), mpiops.chunks * self.n_subchunks)]
        # file tags don't have spaces
        if band_tags:
            file_tags = ["_".join(k.lower().split()) for k in band_tags]
        else:
            file_tags = [str(k) for k in range(self.outbands)]
            band_tags = file_tags

        files = []
        file_names = []

        if mpiops.chunk_index == 0:
            for band in range(self.outbands):
                output_filename = os.path.join(outputdir, name + "_" + file_tags[band] + ".tif")
                f = rasterio.open(output_filename, 'w', driver='GTiff',
                                  width=self.shape[0], height=self.shape[1],
                                  dtype=np.float32, count=1,
                                  crs=crs,
                                  transform=self.A,
                                  nodata=self.nodata_value,
                                  **kwargs
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
        :param subchunk_index: partition number
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
                f.write(data[i:i+1])
        else:
            if mpiops.chunk_index != 0:
                mpiops.comm.Send(image.data, dest=0, tag=1)
                mpiops.comm.Send(image.mask, dest=0, tag=2)
            else:
                for node in range(mpiops.chunks):
                    node = mpiops.chunks - node - 1
                    subindex = mpiops.chunks*subchunk_index + node
                    ystart = self.sub_starts[subindex]
                    yend = self.sub_ends[subindex]  # this is Y
                    if node != 0:
                        data = np.zeros(shape=(self.shape[0], yend - ystart, self.shape[-1]), dtype=np.float32)
                        mask = np.ones(shape=(self.shape[0], yend - ystart, self.shape[-1]), dtype=np.bool)
                        mpiops.comm.Recv(data, source=node, tag=1)
                        mpiops.comm.Recv(mask, source=node, tag=2)
                        data = np.ma.masked_array(data=data, mask=mask, dtype=np.float32, fill_value=self.nodata_value)
                    else:
                        data = image
                    data = np.ma.transpose(data, [2, 1, 0])  # untranspose
                    window = Window(col_off=0, row_off=ystart, width=self.shape[0], height=yend - ystart)
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


def feature_names(config: Config):

    results = []
    for s in config.feature_sets:
        feats = []
        for tif in s.files:
            name = os.path.basename(tif)
            feats.append(name)
        feats.sort()
        results += feats
    return results


def _iterate_sources(f, config: Config):

    results = []
    template_tif = config.prediction_template if config.is_prediction else None
    if config.is_prediction:
        log.info(f"Using prediction template {config.prediction_template}")
    for s in config.feature_sets:
        extracted_chunks = {}
        for tif in s.files:
            name = os.path.abspath(tif)
            image_source = RasterioImageSource(tif)
            x = f(image_source)
            log_missing_percentage(name, x)
            extracted_chunks[name] = x
        extracted_chunks = OrderedDict(sorted(
            extracted_chunks.items(), key=lambda t: t[0]))

        results.append(extracted_chunks)
    return results


def log_missing_percentage(name, x):
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


def image_resolutions(config):
    def f(image_source):
        r = image_source._full_res
        return r

    result = _iterate_sources(f, config)
    return result


def image_subchunks(subchunk_index, config: Config):
    """This is used in prediction only"""

    def f(image_source: RasterioImageSource):
        if config.is_prediction and config.prediction_template is not None:
            template_source = RasterioImageSource(config.prediction_template)
        else:
            template_source = None
        r = features.extract_subchunks(image_source, subchunk_index, config.n_subchunks, config.patchsize,
                                       template_source=template_source)
        return r
    result = _iterate_sources(f, config)
    return result


def extract_intersected_features(image_source: RasterioImageSource, targets: Targets, config: Config):
    othervals = targets.fields
    assert config.intersected_features[Path(image_source.filename).name] in othervals.keys()
    x = othervals[config.intersected_features[Path(image_source.filename).name]]
    x = np.ma.MaskedArray(x, mask=False)
    return x[:, np.newaxis, np.newaxis, np.newaxis]


def image_feature_sets(targets, config: Config):
    def f(image_source):
        if config.intersected_features:
            r = extract_intersected_features(image_source, targets, config)
        else:
            r = features.extract_features(image_source, targets,
                                          config.n_subchunks, config.patchsize)
        return r
    result = _iterate_sources(f, config)
    return result


def semisupervised_feature_sets(targets, config: Config):

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
            rr = r[np.random.rand(r.shape[0]) < frac]
            log.info(f"sampled {rr.shape[0]} from max possible {r.shape[0]}")
        return rr
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
        plt.ylabel(measure)
        plt.savefig('{}.png'.format(measure))

    # Write the results out to a file
    with open(outfile_ranks, 'w') as output_file:
        json.dump(score_listing, output_file, sort_keys=True, indent=4)


def export_model(model, config: Config, learn=True):
    state_dict = {"model": model, "config": config}
    model_file = config.optimised_model_file if (config.optimised_model and not learn) else config.model_file
    with open(model_file, 'wb') as f:
        joblib.dump(state_dict, f)
        log.info(f"Wrote model on disc {model_file}")


# def plot_feature_importance(X, y, xgbmodel: XGBRegressor):
#     all_cols = xgbmodel.feature_importances_
#     non_zero_indices = xgbmodel.feature_importances_ >= 0.001
#     non_zero_cols = X.columns[non_zero_indices]
#     non_zero_importances = xgbmodel.feature_importances_[non_zero_indices]
#     sorted_non_zero_indices = non_zero_importances.argsort()
#     plt.barh(non_zero_cols[sorted_non_zero_indices], non_zero_importances[sorted_non_zero_indices])
#     plt.xlabel("Xgboost Feature Importance")


def export_cluster_model(model, config: Config):
    state_dict = {"model": model, "config": config}
    with open(config.model_file, 'wb') as f:
        pickle.dump(state_dict, f)


class CrossvalInfo:
    def __init__(self, scores, y_true, y_pred, weight, lon_lat, classification):
        self.scores = scores
        self.y_true = y_true
        self.weight = weight
        self.y_pred = y_pred
        self.lon_lat = lon_lat
        self.classification = classification


def export_crossval(crossval_output: CrossvalInfo, config):
    outfile_scores = os.path.join(config.output_dir, config.name + "_scores.json")

    scores = output_json(crossval_output.scores, outfile_scores)

    outfile_results = os.path.join(config.output_dir,
                                   config.name + "_results.hdf5")
    with hdf.open_file(outfile_results, 'w') as f:
        for fld, v in crossval_output.y_pred.items():
            label = _make_valid_array_name(fld)
            if isinstance(v.data, memoryview):
                data = np.array(v.data)
            else:
                data = v.data
            f.create_array("/", label, obj=data)
            if hasattr(v, 'mask'):
                f.create_array("/", label + "_mask", obj=v.mask)
        f.create_array("/", "y_true", obj=crossval_output.y_true)
        if config.weighted_model:
            f.create_array("/", "weight", obj=crossval_output.weight)
        f.create_array("/", "lon_lat", obj=crossval_output.lon_lat)

    if not crossval_output.classification:
        export_validation_scatter_plot_and_validation_csv(outfile_results, config, scores)


def output_json(scores: dict, output_json: Union[str, Path]):
    # Make sure we convert numpy arrays to lists
    scores = {s: v if np.isscalar(v) else v.tolist()
              for s, v in scores.items()}
    with open(output_json, 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)
    return scores


def _make_valid_array_name(label):
    label = "_".join(label.split())
    label = ''.join(filter(str.isalnum, label))  # alphanum only
    if label[0].isdigit():
        label = '_' + label
    return label


def export_validation_scatter_plot_and_validation_csv(outfile_results, config: Config, scores):
    true_vs_pred = os.path.join(config.output_dir,
                                config.name + "_results.csv")
    true_vs_pred_shp = os.path.join(config.output_dir, config.name + "_results.shp")
    true_vs_pred_plot = os.path.join(config.output_dir,
                                     config.name + "_results.png")
    with hdf.open_file(outfile_results, 'r') as f:
        prediction = f.get_node("/", "Prediction").read()
        y_true = f.get_node("/", "y_true").read()
        lon_lat = f.get_node("/", "lon_lat").read()
        to_text = [y_true[:, np.newaxis], prediction[:, np.newaxis]]
        cols = ['y_true', 'y_pred']
        if config.weighted_model:
            weight = f.get_node("/", "weight").read()
            to_text.append(weight[:, np.newaxis])
            cols.append('weight')

        if 'transformedpredict' in f.root:
            transformed_predict = f.get_node("/", "transformedpredict").read()
            to_text.append(transformed_predict[:, np.newaxis])
            cols.append('y_transformed')
        to_text.append(lon_lat)
        cols += ['lon', 'lat']
        X = np.hstack(to_text)
        np.savetxt(true_vs_pred, X=X, delimiter=',',
                   fmt='%.8e',
                   header=','.join(cols),
                   comments='')
        df = pd.DataFrame(X, columns=cols)
        df['diff'] = df['y_true'] - df['y_pred']
        gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df['lon'], y=df['lat'])).to_file(true_vs_pred_shp)
        log.info(f"saved output in shapefile {true_vs_pred_shp}")

        plt.figure()
        plt.scatter(y_true, prediction, label='True vs Prediction')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 color='r', linewidth=2, label='One to One Line')
        plt.legend(loc='upper left')

        plt.title('true vs prediction')
        plt.xlabel('True')
        plt.ylabel('Prediction')
        display_score = ['r2_score', 'lins_ccc']
        score_sring = ''
        for k in display_score:
            score_sring += '{}={:0.2f}\n'.format(k, scores[k])

        plt.text(y_true.min() + (y_true.max() - y_true.min())/20,
                 y_true.min() + (y_true.max() - y_true.min())*3/4,
                 score_sring)
        plt.savefig(true_vs_pred_plot)


def resample(input_tif, output_tif, ratio, resampling="average"):
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
    src.close()
    run(
        f"gdalwarp {input_tif} {output_tif} -tr {src.res[0]*ratio} {src.res[1]*ratio} "
        f"-wm 2034 -r {resampling} -overwrite",
        shell=True
    )
