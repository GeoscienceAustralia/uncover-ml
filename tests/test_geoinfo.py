import unittest
import os
import glob
import numpy as np
from osgeo import gdal
from preprocessing import geoinfo


UNCOVER = os.environ['UNCOVER']  # points to the uncover-ml directory


class Geoinfo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mocks = os.path.join(UNCOVER, 'preprocessing', 'mocks')
        cls.tifs = glob.glob(os.path.join(mocks, '*.tif'))

    def test_gdal_vs_numpy(self):
        for t in self.tifs:
            ds = gdal.Open(t)
            n_list = geoinfo.numpy_band_stats(ds, t, 1)
            g_list = geoinfo.band_stats(ds, t, 1)
            print(n_list)
            print(g_list)
            self.assertEqual(n_list[0], g_list[0])
            self.assertEqual(n_list[-2], g_list[-2])

            np.testing.assert_almost_equal(
                np.array([float(t) for t in n_list[1:-2]]),
                np.array([float(t) for t in g_list[1:-2]]),
                decimal=4
            )


if __name__ == '__main__':
    unittest.main()
