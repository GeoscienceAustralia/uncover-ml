import numpy as np

from uncoverml import patch


def test_grid_patch(make_patch):

    timg, psize, pstride, tpatch, tx, ty = make_patch

    patcher = patch.grid_patches(timg, psize, pstride)
    patches = [(p, x, y) for p, x, y in patcher]
    ps, cxs, cys = zip(*patches)

    assert np.all(np.array(ps) == tpatch)
    assert np.all(np.array(cxs) == tx)
    assert np.all(np.array(cys) == ty)
