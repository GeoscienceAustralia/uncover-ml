import numpy as np
import numpy.ma as ma
import pytest
from uncoverml.filtering import (
    pad2, fwd_filter, inv_filter, kernel_impute, sensor_footprint
)


@pytest.fixture
def sample_image():
    data = np.random.rand(32, 32, 3)
    mask2d = np.zeros((32, 32), dtype=bool)
    mask2d[10:15, 10:15] = True
    mask3d = np.repeat(mask2d[:, :, np.newaxis], 3, axis=2)
    return ma.MaskedArray(data, mask=mask3d)


@pytest.fixture
def sample_kernel():
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2))
    return kernel / kernel.sum()


def test_pad2_shapes(sample_image):
    padded = pad2(sample_image)
    expected_shape = (sample_image.shape[0]*2 - 1, sample_image.shape[1]*2 - 1, sample_image.shape[2])
    assert padded.shape == expected_shape


def test_fwd_filter_runs(sample_image, sample_kernel):
    result = fwd_filter(sample_image, sample_kernel)
    assert isinstance(result, ma.MaskedArray)
    assert result.shape == sample_image.shape


def test_kernel_impute_fills_mask(sample_image, sample_kernel):
    imputed = kernel_impute(sample_image, sample_kernel)
    assert np.all(imputed.mask == False)
    assert np.all(np.isfinite(imputed.data))


def test_inv_filter_restores_shape(sample_image, sample_kernel):
    imputed = kernel_impute(sample_image, sample_kernel)
    result = inv_filter(imputed, sample_kernel, noise=0.01)
    assert isinstance(result, ma.MaskedArray)
    assert result.shape == sample_image.shape


def test_sensor_footprint_shape_and_scale():
    footprint = sensor_footprint(img_w=10, img_h=10, res_x=1.0, res_y=1.0, height=5.0, mu_air=0.01)
    assert footprint.shape == (19, 19)
    assert np.isclose(footprint.max(), 1.0)


def test_fwd_and_inv_consistency(sample_image, sample_kernel):
    imputed = kernel_impute(sample_image, sample_kernel)
    blurred = fwd_filter(imputed, sample_kernel)
    deblurred = inv_filter(blurred, sample_kernel, noise=0.01)
    diff = np.abs(imputed.data - deblurred.data)
    assert np.mean(diff) < 0.3
