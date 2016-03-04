import pytest
import numpy as np


@pytest.fixture
def make_patch():
    timg = np.vstack([np.arange(1, 5),
                      np.arange(5, 9),
                      np.arange(9, 13),
                      np.arange(13, 17)])
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
