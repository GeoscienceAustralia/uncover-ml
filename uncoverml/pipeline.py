import pickle
import logging

import uncoverml.defaults as df
from uncoverml import mpiops
from uncoverml import patch
from uncoverml import stats
from uncoverml import geoio


log = logging.getLogger(__name__)


class PickledSettings:
    def from_file(settings_file):
        s = pickle.load(settings_file)
        return s

    def save(self, settings_file):
        with open(settings_file, 'wb') as f:
            pickle.dump(self, f)


class ExtractSettings(PickledSettings):

    def __init__(self, onehot, x_sets, patchsize):
        self.onehot = onehot
        self.x_sets = x_sets
        self.patchsize = patchsize


class ComposeSettings(PickledSettings):

    def __init__(self, impute, transform, featurefraction, impute_mean, mean,
                 sd, eigvals, eigvecs):
        self.impute = impute
        self.transform = transform
        self.featurefraction = featurefraction
        self.impute_mean = impute_mean
        self.mean = mean
        self.sd = sd
        self.eigvals = eigvals
        self.eigvecs = eigvecs


def extract_transform(x, x_sets):
    x = x.reshape(x.shape[0], -1)
    if x_sets:
        x = stats.one_hot(x, x_sets)
    x = x.astype(float)
    return x


def extract_features(settings, target_infile, geotiff_infile, hdf_outfile):

    # Compute the effective sampled resolution accounting for patchsize
    image_source = geoio.RasterioImageSource(geotiff_infile)
    full_image = geoio.Image(image_source)
    eff_shape = full_image.patched_shape(settings.patchsize)
    eff_bbox = full_image.patched_bbox(settings.patchsize)

    image = geoio.Image(image_source, mpiops.chunk_index,
                        mpiops.chunks, settings.patchsize)

    x = patch.load(image, settings.patchsize, target_infile)

    if settings.onehot and not settings.x_sets:
        settings.x_sets = mpiops.compute_unique_values(x, df.max_onehot_dims)

    if x is not None:
        x = extract_transform(x, settings.x_sets)

    geoio.output_features(x, hdf_outfile, shape=eff_shape, bbox=eff_bbox)

    return settings


def compose_features(settings, hdf_infiles, hdf_outfile):

    # verify the files are all present
    filename_dict = geoio.files_by_chunk(hdf_infiles)

    # Get attribs if they exist
    eff_shape, eff_bbox = geoio.load_attributes(filename_dict)
    chunk_files = filename_dict[mpiops.chunk_index]
    x = geoio.load_and_cat(chunk_files)

    x, settings = mpiops.compose_transform(x, settings)

    geoio.output_features(x, hdf_outfile, shape=eff_shape, bbox=eff_bbox)

    return settings
