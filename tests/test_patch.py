import pytest
import numpy as np

from uncoverml import patch


@pytest.fixture
def make_patch_31():
    pwidth = 1
    pstride = 1

    # Test output patches, patch centres
    tpatch = np.array([[[1, 2, 3],
                        [5, 6, 7],
                        [9, 10, 11]],
                       [[2, 3, 4],
                        [6, 7, 8],
                        [10, 11, 12]],
                       [[5, 6, 7],
                        [9, 10, 11],
                        [13, 14, 15]],
                       [[6, 7, 8],
                        [10, 11, 12],
                        [14, 15, 16]]])

    tx = np.array([1, 1, 2, 2])
    ty = np.array([1, 2, 1, 2])

    return timg, pwidth, pstride, tpatch, tx, ty


@pytest.fixture
def make_patch_11():
    pwidth = 0
    pstride = 1

    # Test output patches, patch centres
    tpatch = np.array([[timg.flatten()]]).T

    tx, ty = [g.flatten() for g in np.meshgrid(np.arange(3), np.arange(3))]

    return timg, pwidth, pstride, tpatch, tx, ty


@pytest.fixture
def make_patch_12():
    pwidth = 0
    pstride = 2

    # Test output patches, patch centres
    tpatch = np.array([[[1]],
                       [[3]],
                       [[9]],
                       [[11]]])

    tx = np.array([0, 0, 2, 2])
    ty = np.array([0, 2, 0, 2])

    return timg, pwidth, pstride, tpatch, tx, ty



def test_grid_patch(make_multi_patch):

    timg, pwidth, pstride, tpatch, tx, ty = make_multi_patch

    # patches = [p, x, y) for p, x, y
    #            in patch.grid_patches(timg, pwidth)]
    # ps, cxs, cys = zip(*patches)
    patches = np.array(list(patch.grid_patches(timg, pwidth, pstride)))

    assert np.allclose(patches, tpatch)


def test_point_patches(make_points):

    timg, pwidth, points, tpatch = make_points

    patches = np.array(list(patch.point_patches(timg, pwidth, points)))

    assert np.allclose(patches, tpatch)
