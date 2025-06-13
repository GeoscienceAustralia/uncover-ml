import numpy as np
import numpy.ma as ma
from types import SimpleNamespace
from uncoverml import patch


class TestImage:
    def __init__(self, data, mask):
        self._data = ma.masked_array(data=data, mask=mask)
    def data(self):
        return self._data
    def in_bounds(self, lonlats):
        return np.array([True] * len(lonlats))
    def lonlat2pix(self, lonlats):
        return np.clip(lonlats.astype(int), 0, self._data.shape[0] - 1)


def test_grid_patch(make_multi_patch):

    timg, pwidth, tpatch, tx, ty = make_multi_patch

    patches = patch.grid_patches(timg, pwidth)

    assert np.allclose(patches, tpatch)


def test_point_patches(make_points):

    timg, pwidth, points, tpatch = make_points

    patches = np.array(list(patch.point_patches(timg, pwidth, points)))

    assert np.allclose(patches, tpatch)


def test_image_to_data():
    img = TestImage(np.random.rand(5, 5, 2).astype(np.float32), np.zeros((5, 5, 2), dtype=bool))
    data, mask, dtype = patch._image_to_data(img)
    assert isinstance(data, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert dtype == np.float32


def test_all_patches():
    img = TestImage(np.random.rand(5, 5, 2).astype(np.float32), np.zeros((5, 5, 2), dtype=bool))
    patchsize = 1
    result = patch.all_patches(img, patchsize)
    assert isinstance(result, ma.MaskedArray)


def test_patches_at_target():
    img = TestImage(np.random.rand(5, 5, 2).astype(np.float32), np.zeros((5, 5, 2), dtype=bool))
    patchsize = 1
    targets = SimpleNamespace(positions=np.array([[1, 1], [2, 2], [3, 3]]))
    result = patch.patches_at_target(img, patchsize, targets)
    assert isinstance(result, ma.MaskedArray)
    assert result.shape[1] == result.shape[2] == 2 * patchsize + 1
