import os
import glob
import numpy as np
from osgeo import gdal
from preprocessing import geoinfo

def test_gdal_vs_numpy(mock_files):
    for t in list(mock_files.values()):
        ds = gdal.Open(t)
        n_list = geoinfo.numpy_band_stats(ds, t, 1)
        g_list = geoinfo.band_stats(ds, t, 1)
        assert n_list[0] == g_list[0]
        assert n_list[-3] == g_list[-3]
        assert n_list[-2] == g_list[-2]

        np.testing.assert_almost_equal(
            np.array([float(t) for t in n_list[1:-3]]),
            np.array([float(t) for t in g_list[1:-3]]),
            decimal=4
        )
