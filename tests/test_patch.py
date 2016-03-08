import numpy as np

from uncoverml import patch


def test_grid_patch(make_multi_patch):

    timg, pwidth, pstride, tpatch, ty, tx = make_multi_patch

    patches = [(p, y, x) for p, y, x
               in patch.grid_patches(timg, pwidth, pstride)]
    ps, cys, cxs = zip(*patches)

    assert np.all(np.array(ps) == tpatch)
    assert np.all(np.array(cxs) == tx)
    assert np.all(np.array(cys) == ty)


def test_image_windows(make_multi_patch):

    timg, pwidth, pstride, tpatch, ty, tx = make_multi_patch

    slices = patch.image_windows(timg.shape, 4, pwidth, pstride)

    patches = []
    for sl in slices:
        offset = (sl[0].start, sl[1].start)
        patches.extend([(p, y, x) for p, y, x
                        in patch.grid_patches(timg[sl],
                                              pwidth,
                                              pstride,
                                              centreoffset=offset)
                        ])

    ps, cys, cxs = zip(*patches)

    assert np.all(np.array(ps) == tpatch)
    assert np.all(np.array(cxs) == tx)
    assert np.all(np.array(cys) == ty)
