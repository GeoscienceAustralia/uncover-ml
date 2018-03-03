import numpy as np
from numpy import nan
from os.path import join, basename
import tempfile
from osgeo import gdal
import shutil
from itertools import product
from preprocessing import raster_average


class TestRasterAverage:

    @classmethod
    def setup_class(cls):
        cls.data = np.array([[1.0, 1.2, 1.4, 1.6, 1.8],
                             [1.0, 1.2, 1.4, 1.6, 1.8],
                             [1.0, 1.2, 1.4, 1.6, 1.8],
                             [1.0, 1.2, 1.4, 1.6, 1.8],
                             [1.0, 1.2, 1.4, 1.6, 1.8]])
        cls.expected_average_2 = np.array([[1.0, 1.1, 1.3, 1.5, 1.7],
                                           [1.0, 1.1, 1.3, 1.5, 1.7],
                                           [1.0, 1.1, 1.3, 1.5, 1.7],
                                           [1.0, 1.1, 1.3, 1.5, 1.7],
                                           [1.0, 1.1, 1.3, 1.5, 1.7]])

        cls.data_rand = np.random.rand(5, 5)

    def test_average_size2(self):
        averaged_data = raster_average.filter_data(self.data,
                                                   size=2)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_2)

    def test_average_size3(self):
        averaged_data = raster_average.filter_data(self.data,
                                                   size=3)
        np.testing.assert_array_almost_equal(averaged_data[:, 1:-1],
                                             self.data[:, 1:-1])


class TestUniformFilterWithNoData:

    @classmethod
    def setup_class(cls):
        cls.data = np.array([[1000.0, 1.2, 1.4, 1.6, 1.8],
                             [1000.0, 1.2, 1.4, 1.6, 1.8],
                             [1.0, 1.2, 1000.0, 1.6, 1.8],
                             [2.0, 1.2, 1.4, 1.6, 1.8],
                             [1000.0, 1.2, 1.4, 1.6, 1.8]])

        cls.expected_average_3 = np.array([[nan, 1.2, 1.3, 1.4, 1.6],
                                           [nan, 1.2, 1.3, 1.4, 1.6],
                                           [1.0, 1.15, 1.23333333, 1.4, 1.625],
                                           [1.5, 1.32, 1.34285714, 1.4, 1.625],
                                           [1.5, 1.32, 1.34285714, 1.4, 1.625]])

        cls.expected_average_5 = np.array([[nan, 1.2, 1.3, 1.4, 1.5],
                                           [nan, 1.2, 1.3, 1.4, 1.5],
                                           [1.0, 1.15, 1.23333333, 1.35555556,
                                            1.46666667],
                                           [1.5, 1.3, 1.33333333, 1.41538462,
                                            1.50588235],
                                           [1.5, 1.28571429, 1.32727273,
                                            1.4125, 1.5047619]])

    def test_average_size3(self):
        averaged_data = raster_average.filter_uniform_filter(
            self.data, size=3, no_data_val=1000.0, func=np.nanmean)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_3)

    def test_average_size5(self):
        averaged_data = raster_average.filter_uniform_filter(
            self.data, size=5, no_data_val=1000.0, func=np.nanmean)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_5)


class TestFilterCenterWithNoDataAndMask:

    @classmethod
    def setup_class(cls):
        cls.data = np.array([[1000.0, 1.2, 1.4, 1.6, 1.8],
                             [1000.0, 1.2, 1.4, 1.6, 1.8],
                             [1.0, 1.2, 1000.0, 1.6, 1.8],
                             [2.0, 1.2, 1.4, 1.6, 1.8],
                             [1000.0, 1.2, 1.4, 1.6, 1.8]])

        cls.expected_average_3 = np.array(
            [[1000.0, 1.3, 1.4, 1.6, 1.7],
             [1000.0, 1.23333333, 1.4, 1.625, 1.7],
             [1.32, 1.34285714, 1000.0, 1.625, 1.7],
             [1.32, 1.34285714, 1.4, 1.625, 1.7],
             [1000.0, 1.44, 1.4, 1.6, 1.7]])

        cls.expected_average_5 = np.array(
            [[1000.0, 1.35555556, 1.46666667, 1.50909091, 1.625],
             [1000.0, 1.41538462, 1.50588235, 1.50666667, 1.61818182],
             [1.32727273, 1.4125, 1000.0, 1.50526316, 1.61428571],
             [1.33333333, 1.41538462, 1.50588235, 1.50666667, 1.61818182],
             [1000.0, 1.42, 1.50769231, 1.50909091, 1.625]]
            )

        cls.data_5x4 = np.array([[1000.0, 1.2, 1.4, 1.6],
                                 [1000.0, 1.2, 1.4, 1.6],
                                 [1.0, 1.2, 1000.0, 1.6],
                                 [2.0, 1.2, 1.4, 1.6],
                                 [1000.0, 1.2, 1.4, 1.6]])
        cls.expected_average_3_5x4 = np.array([[1000.0, 1.3, 1.4, 1.5],
                                               [1000.0, 1.23333333, 1.4, 1.52],
                                               [1.32, 1.34285714, 1000.0, 1.52],
                                               [1.32, 1.34285714, 1.4, 1.52],
                                               [1000.0, 1.44, 1.4, 1.5]])

    def test_average_size3(self):
        averaged_data = raster_average.filter_center(
            self.data, size=3, no_data_val=1000.0, func=np.nanmean,
            mask_no_data=True)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_3)

    def test_average_size3_5x4(self):
        averaged_data = raster_average.filter_center(
            self.data_5x4, size=3, no_data_val=1000.0, func=np.nanmean,
            mask_no_data=True)

        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_3_5x4)

    def test_average_size5(self):
        averaged_data = raster_average.filter_center(
            self.data, size=5, no_data_val=1000.0, func=np.nanmean,
            mask_no_data=True)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_5)


class TestFilterCenterWithNoData:

    @classmethod
    def setup_class(cls):
        cls.data = np.array([[1000.0, 1.2, 1.4, 1.6, 1.8],
                             [1000.0, 1.2, 1.4, 1.6, 1.8],
                             [1.0, 1.2, 1000.0, 1.6, 1.8],
                             [2.0, 1.2, 1.4, 1.6, 1.8],
                             [1000.0, 1.2, 1.4, 1.6, 1.8]])

        cls.expected_average_3 = np.array([[1.2, 1.3, 1.4, 1.6, 1.7],
                                           [1.15, 1.23333333, 1.4, 1.625, 1.7],
                                           [1.32, 1.34285714, 1.4, 1.625, 1.7],
                                           [1.32, 1.34285714, 1.4, 1.625, 1.7],
                                           [1.46666667, 1.44, 1.4, 1.6, 1.7]])

        cls.expected_average_5 = np.array(
            [[1.23333333, 1.35555556, 1.46666667, 1.50909091, 1.625],
             [1.33333333, 1.41538462, 1.50588235, 1.50666667, 1.61818182],
             [1.32727273, 1.4125, 1.5047619, 1.50526316, 1.61428571],
             [1.33333333, 1.41538462, 1.50588235, 1.50666667, 1.61818182],
             [1.34285714, 1.42, 1.50769231, 1.50909091, 1.625]]
            )

        cls.data_5x4 = np.array([[1000.0, 1.2, 1.4, 1.6],
                                 [1000.0, 1.2, 1.4, 1.6],
                                 [1.0, 1.2, 1000.0, 1.6],
                                 [2.0, 1.2, 1.4, 1.6],
                                 [1000.0, 1.2, 1.4, 1.6]])
        cls.expected_average_3_5x4 = np.array([[1.2, 1.3, 1.4, 1.5],
                                              [1.15, 1.23333333, 1.4, 1.52],
                                              [1.32, 1.34285714, 1.4, 1.52],
                                              [1.32, 1.34285714, 1.4, 1.52],
                                              [1.46666667, 1.44, 1.4, 1.5]])

    def test_average_size3(self):
        averaged_data = raster_average.filter_center(
            self.data, size=3, no_data_val=1000.0, func=np.nanmean)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_3)

    def test_average_size3_5x4(self):
        averaged_data = raster_average.filter_center(
            self.data_5x4, size=3, no_data_val=1000.0, func=np.nanmean)

        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_3_5x4)

    def test_average_size5(self):
        averaged_data = raster_average.filter_center(
            self.data, size=5, no_data_val=1000.0, func=np.nanmean)
        np.testing.assert_array_almost_equal(averaged_data,
                                             self.expected_average_5)


def test_mpi_vs_serial(uncover):
    std2000 = join(uncover, 'preprocessing', 'mocks', 'std2000.tif')
    test_tif = std2000

    tmpdir1 = tempfile.mkdtemp()
    tmpdir2 = tempfile.mkdtemp()

    for n, partitions in product(range(1, 3), range(2, 100, 10)):
        size = 2 * n + 1
        raster_average.treat_file(test_tif,
                                  out_dir=tmpdir1,
                                  size=size,
                                  func='nanmean',
                                  partitions=partitions)
        arr1 = gdal.Open(
            join(tmpdir1, basename(test_tif))).ReadAsArray()

        raster_average.treat_file(test_tif,
                                  out_dir=tmpdir2,
                                  size=size,
                                  func='nanmean',
                                  partitions=partitions)
        arr2 = gdal.Open(join(tmpdir2,
                              basename(test_tif))).ReadAsArray()
        np.testing.assert_array_almost_equal(arr1, arr2)

    shutil.rmtree(tmpdir2)
    shutil.rmtree(tmpdir1)


