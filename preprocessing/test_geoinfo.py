import unittest
import os
import glob
import numpy as np
from preprocessing import geoinfo


UNCOVER = os.environ['UNCOVER']  # points to the uncover-ml directory


class Geoinfo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mocks = os.path.join(UNCOVER, 'preprocessing', 'mocks')
        cls.tifs = glob.glob(os.path.join(mocks, '*.tif'))
        print(cls.tifs)

    def test_gdal_vs_numpy(self):
        for t in self.tifs:
            n_list = geoinfo.get_numpy_stats(t)
            g_list = geoinfo.get_stats(t)
            self.assertEqual(n_list[0], g_list[0])
            np.testing.assert_almost_equal(
                np.array([float(t) for t in n_list[1:]]),
                np.array([float(t) for t in g_list[1:]]),
                decimal=4
            )


if __name__ == '__main__':
    unittest.main()
