import pytest
import numpy as np

timg = np.reshape(np.arange(1, 17), (4, 4))


@pytest.fixture
def make_patch_31():
    psize = 3
    pstride = 1

    # Test output patches, patch centres
    tpatch = np.array([[1, 2, 3, 5, 6, 7, 9, 10, 11],
                       [2, 3, 4, 6, 7, 8, 10, 11, 12],
                       [5, 6, 7, 9, 10, 11, 13, 14, 15],
                       [6, 7, 8, 10, 11, 12, 14, 15, 16]])

    tx = np.array([1, 2, 1, 2])
    ty = np.array([1, 1, 2, 2])

    return timg, psize, pstride, tpatch, tx, ty


@pytest.fixture
def make_patch_22():
    psize = 2
    pstride = 2

    # Test output patches, patch centres
    tpatch = np.array([[1, 2, 5, 6],
                       [3, 4, 7, 8],
                       [9, 10, 13, 14],
                       [11, 12, 15, 16]])

    tx = np.array([0.5, 2.5, 0.5, 2.5])
    ty = np.array([0.5, 0.5, 2.5, 2.5])

    return timg, psize, pstride, tpatch, tx, ty


@pytest.fixture(params=[make_patch_22, make_patch_31])
def make_multi_patch(request):
    return request.param()
