import pytest
import numpy as np

timg = np.reshape(np.arange(1, 17), (4, 4))


@pytest.fixture
def make_patch_31():
    pwidth = 1
    pstride = 1

    # Test output patches, patch centres
    tpatch = np.array([[1, 2, 3, 5, 6, 7, 9, 10, 11],
                       [2, 3, 4, 6, 7, 8, 10, 11, 12],
                       [5, 6, 7, 9, 10, 11, 13, 14, 15],
                       [6, 7, 8, 10, 11, 12, 14, 15, 16]])

    tx = np.array([1, 2, 1, 2])
    ty = np.array([1, 1, 2, 2])

    return timg, pwidth, pstride, tpatch, ty, tx


@pytest.fixture
def make_patch_32():
    pwidth = 1
    pstride = 2

    # Test output patches, patch centres
    tpatch = np.array([[1, 2, 3, 5, 6, 7, 9, 10, 11]])

    tx = np.array([1])
    ty = np.array([1])

    return timg, pwidth, pstride, tpatch, ty, tx


@pytest.fixture
def make_points():
    pwidth = 1
    points = np.array([[1, 1], [2, 1], [2, 2]])

    tpatch = np.array([[1, 2, 3, 5, 6, 7, 9, 10, 11],
                       [5, 6, 7, 9, 10, 11, 13, 14, 15],
                       [6, 7, 8, 10, 11, 12, 14, 15, 16]])

    return timg, pwidth, points, tpatch


@pytest.fixture(params=[make_patch_31, make_patch_32])
def make_multi_patch(request):
    return request.param()
