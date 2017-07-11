import unittest
import os
import numpy as np
from numpy import nan
from os.path import join, basename
from osgeo import gdal
import shutil
from itertools import product

from preprocessing import raster_average


simple_data = np.array([[1.0, 1.2, 1.4, 1.6, 1.8],
                      [1.0, 1.2, 1.4, 1.6, 1.8],
                      [1.0, 1.2, 1.4, 1.6, 1.8],
                      [1.0, 1.2, 1.4, 1.6, 1.8],
                      [1.0, 1.2, 1.4, 1.6, 1.8]])

data = np.array([[1000.0, 1.2, 1.4, 1.6, 1.8],
                  [1000.0, 1.2, 1.4, 1.6, 1.8],
                  [1.0, 1.2, 1000.0, 1.6, 1.8],
                  [2.0, 1.2, 1.4, 1.6, 1.8],
                  [1000.0, 1.2, 1.4, 1.6, 1.8]])

filled_data = np.array([[1000.0, 1.2, 1.4, 1.6, 1.8],
                      [1000.0, 1.2, 1.4, 1.6, 1.8],
                      [1.0, 1.2, 1000.0, 1.6, 1.8],
                      [2.0, 1.2, 1.4, 1.6, 1.8],
                      [1000.0, 1.2, 1.4, 1.6, 1.8]])


data_5x4 = np.array([[1000.0, 1.2, 1.4, 1.6],
                      [1000.0, 1.2, 1.4, 1.6],
                      [1.0, 1.2, 1000.0, 1.6],
                      [2.0, 1.2, 1.4, 1.6],
                      [1000.0, 1.2, 1.4, 1.6]])

uniform_data = np.array([[1000.0, 1.2, 1.4, 1.6, 1.8],
                          [1000.0, 1.2, 1.4, 1.6, 1.8],
                          [1.0, 1.2, 1000.0, 1.6, 1.8],
                          [2.0, 1.2, 1.4, 1.6, 1.8],
                          [1000.0, 1.2, 1.4, 1.6, 1.8]])


def test_average_size2():
    expected_average_2 = np.array([[1.0, 1.1, 1.3, 1.5, 1.7],
                                        [1.0, 1.1, 1.3, 1.5, 1.7],
                                        [1.0, 1.1, 1.3, 1.5, 1.7],
                                        [1.0, 1.1, 1.3, 1.5, 1.7],
                                        [1.0, 1.1, 1.3, 1.5, 1.7]])
    averaged_data = raster_average.filter_data(simple_data, size=2)
    assert np.allclose(averaged_data,expected_average_2)

def test_average_size3():
    averaged_data = raster_average.filter_data(simple_data, size=3)
    assert np.allclose(averaged_data[:, 1:-1], simple_data[:, 1:-1])



def test_uniform_size3():
    expected_average_3 = np.array([[nan, 1.2, 1.3, 1.4, 1.6],
                                        [nan, 1.2, 1.3, 1.4, 1.6],
                                        [1.0, 1.15, 1.23333333, 1.4, 1.625],
                                        [1.5, 1.32, 1.34285714, 1.4, 1.625],
                                        [1.5, 1.32, 1.34285714, 1.4,
                                         1.625]])

    averaged_data = raster_average.filter_uniform_filter(
        uniform_data, size=3, no_data_val=1000.0, func=np.nanmean)
    np.testing.assert_array_almost_equal(averaged_data, expected_average_3)

def test_uniform_size5():
    expected_average_5 = np.array([[nan, 1.2, 1.3, 1.4, 1.5],
                                    [nan, 1.2, 1.3, 1.4, 1.5],
                                    [1.0, 1.15, 1.23333333, 1.35555556,
                                     1.46666667],
                                   [1.5, 1.3, 1.33333333, 1.41538462,
                                    1.50588235],
                                   [1.5, 1.28571429, 1.32727273,
                                    1.4125, 1.5047619]])

    averaged_data = raster_average.filter_uniform_filter(
        uniform_data, size=5, no_data_val=1000.0, func=np.nanmean)
    np.testing.assert_array_almost_equal(averaged_data, expected_average_5)



def test_centre_size3():
    expected_average_3 = np.array([[1000.0, 1.3, 1.4, 1.6, 1.7],
                                    [1000.0, 1.23333333, 1.4, 1.625, 1.7],
                                    [1.32, 1.34285714, 1000.0, 1.625, 1.7],
                                    [1.32, 1.34285714, 1.4, 1.625, 1.7],
                                    [1000.0, 1.44, 1.4, 1.6, 1.7]])

    averaged_data = raster_average.filter_center(
        data, size=3, no_data_val=1000.0, func=np.nanmean,
        mask_no_data=True)
    assert np.allclose(averaged_data, expected_average_3)

def test_centre_size3_5x4():
    expected_average_3_5x4 = np.array([[1000.0, 1.3, 1.4, 1.5],
                                        [1000.0, 1.23333333, 1.4, 1.52],
                                        [1.32, 1.34285714, 1000.0, 1.52],
                                        [1.32, 1.34285714, 1.4, 1.52],
                                        [1000.0, 1.44, 1.4, 1.5]])

    averaged_data = raster_average.filter_center(
        data_5x4, size=3, no_data_val=1000.0, func=np.nanmean,
        mask_no_data=True)
    assert np.allclose(averaged_data, expected_average_3_5x4)


def test_centre_size5():
    expected_average_5 = np.array(
        [[1000.0, 1.35555556, 1.46666667, 1.50909091, 1.625],
         [1000.0, 1.41538462, 1.50588235, 1.50666667, 1.61818182],
         [1.32727273, 1.4125, 1000.0, 1.50526316, 1.61428571],
         [1.33333333, 1.41538462, 1.50588235, 1.50666667, 1.61818182],
         [1000.0, 1.42, 1.50769231, 1.50909091, 1.625]]
        )
    averaged_data = raster_average.filter_center(
                filled_data, size=5, no_data_val=1000.0, func=np.nanmean,
                mask_no_data=True)
    np.testing.assert_array_almost_equal(averaged_data, expected_average_5)




def test_nodata_size3():
    expected_average_3 = np.array([[1.2, 1.3, 1.4, 1.6, 1.7],
                                        [1.15, 1.23333333, 1.4, 1.625, 1.7],
                                        [1.32, 1.34285714, 1.4, 1.625, 1.7],
                                        [1.32, 1.34285714, 1.4, 1.625, 1.7],
                                        [1.46666667, 1.44, 1.4, 1.6, 1.7]])

    averaged_data = raster_average.filter_center(
        data, size=3, no_data_val=1000.0, func=np.nanmean)
    assert np.allclose(averaged_data, expected_average_3)

def test_nodata_size3_5x4():
    expected_average_3_5x4 = np.array([[1.2, 1.3, 1.4, 1.5],
                                        [1.15, 1.23333333, 1.4, 1.52],
                                        [1.32, 1.34285714, 1.4, 1.52],
                                        [1.32, 1.34285714, 1.4, 1.52],
                                        [1.46666667, 1.44, 1.4, 1.5]])

    averaged_data = raster_average.filter_center(data_5x4, size=3,
                                                 no_data_val=1000.0,
                                                 func=np.nanmean)
    assert np.allclose(averaged_data, expected_average_3_5x4)

def test_nodata_size5():
    expected_average_5 = np.array(
        [[1.23333333, 1.35555556, 1.46666667, 1.50909091, 1.625],
         [1.33333333, 1.41538462, 1.50588235, 1.50666667, 1.61818182],
         [1.32727273, 1.4125, 1.5047619, 1.50526316, 1.61428571],
         [1.33333333, 1.41538462, 1.50588235, 1.50666667, 1.61818182],
         [1.34285714, 1.42, 1.50769231, 1.50909091, 1.625]]
        )

    averaged_data = raster_average.filter_center(
        data, size=5, no_data_val=1000.0, func=np.nanmean)
    assert np.allclose(averaged_data, expected_average_5)

def test_partition(mock_files, random_dir):
    tmpdir1 = random_dir()
    tmpdir2 = random_dir()
    test_tif = mock_files['std2000']
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

