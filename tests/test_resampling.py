import pytest
import shapefile as shp
import numpy as np
from uncoverml import resampling
from uncoverml import geoio
import tempfile
import shutil


@pytest.fixture(params=range(2, 4))
def rows(request):
    return request.param


@pytest.fixture(params=range(2, 4))
def cols(request):
    return request.param


@pytest.fixture(params=range(2, 5))
def bins(request):
    return request.param


@pytest.fixture(params=range(10, 200, 50))
def samples(request):
    return request.param


def test_resampling_value(shapefile, bins, samples):
    samples = (samples//bins) * bins
    lonlats, filename = shapefile
    random_filename = tempfile.mktemp() + ".shp"
    resampling.resample_by_magnitude(filename,
                                     random_filename,
                                     target_field='lat',
                                     bins=bins,
                                     output_samples=samples,
                                     bootstrap=True
                                     )
    resampled_sf = shp.Reader(random_filename)
    resampled_shapefields = [f[0] for f in resampled_sf.fields[1:]]

    new_coords, new_val, new_othervals = \
        geoio.load_shapefile(random_filename, 'lat')

    assert 'lat' in resampled_shapefields
    assert 'lon' not in resampled_shapefields
    assert np.all((samples, 2) == new_coords.shape)
    assert new_othervals == {}  # only the target is retained after resampling


def test_resampling_spatial(shapefile, rows, cols, samples):
    tiles = rows * cols
    samples = (samples // tiles) * tiles
    lonlats, filename = shapefile
    random_filename = tempfile.mktemp() + ".shp"
    resampling.resample_spatially(filename,
                                  random_filename,
                                  target_field='lat',
                                  rows=rows,
                                  cols=cols,
                                  output_samples=samples,
                                  bootstrap=True
                                  )
    resampled_sf = shp.Reader(random_filename)
    resampled_shapefields = [f[0] for f in resampled_sf.fields[1:]]

    new_coords, new_val, new_othervals = \
        geoio.load_shapefile(random_filename, 'lat')

    assert 'lat' in resampled_shapefields
    assert 'lon' not in resampled_shapefields
    assert np.all((samples, 2) >= new_coords.shape)
    assert new_othervals == {}  # only the target is retained after resampling
