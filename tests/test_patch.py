import numpy as np

from uncoverml import patch


def test_grid_patch(make_multi_patch):

    timg, pwidth, tpatch, tx, ty = make_multi_patch

    patches = patch.grid_patches(timg, pwidth)

    assert np.allclose(patches, tpatch)


def test_point_patches(make_points):

    timg, pwidth, points, tpatch = make_points

    patches = np.array(list(patch.point_patches(timg, pwidth, points)))

    assert np.allclose(patches, tpatch)
