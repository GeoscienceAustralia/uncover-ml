import unittest
import os
import numpy as np
from preprocessing import raster_average

UNCOVER = os.environ['UNCOVER']


class TestRasterAverage(unittest.TestCase):

    def setUp(self):
        self.data = np.array([[1.0, 1.2, 1.4, 1.6, 1.8],
                              [1.0, 1.2, 1.4, 1.6, 1.8],
                              [1.0, 1.2, 1.4, 1.6, 1.8],
                              [1.0, 1.2, 1.4, 1.6, 1.8],
                              [1.0, 1.2, 1.4, 1.6, 1.8]])
        self.expected_average_2 = np.array([[1.0, 1.1, 1.3, 1.5, 1.7],
                                            [1.0, 1.1, 1.3, 1.5, 1.7],
                                            [1.0, 1.1, 1.3, 1.5, 1.7],
                                            [1.0, 1.1, 1.3, 1.5, 1.7],
                                            [1.0, 1.1, 1.3, 1.5, 1.7]])

        self.data_rand = np.random.rand(5, 5)

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


# class TestRasterAverageWithNoData(unittest.TestCase):
#
#     def setUp(self):
#         self.data = np.array([[1000.0, 1.2, 1.4, 1.6, 1.8],
#                               [1000.0, 1.2, 1.4, 1.6, 1.8],
#                               [1000.0, 1.2, 1.4, 1.6, 1.8],
#                               [1000.0, 1.2, 1.4, 1.6, 1.8],
#                               [1000.0, 1.2, 1.4, 1.6, 1.8]])
#         self.expected_average_2 = np.array([[1.0, 1.1, 1.3, 1.5, 1.7],
#                                             [1.0, 1.1, 1.3, 1.5, 1.7],
#                                             [1.0, 1.1, 1.3, 1.5, 1.7],
#                                             [1.0, 1.1, 1.3, 1.5, 1.7],
#                                             [1.0, 1.1, 1.3, 1.5, 1.7]])
#
#         self.data_rand = np.random.rand(5, 5)
#
#     def test_average_size2(self):
#         averaged_data = raster_average.filter_data(self.data,
#                                                    size=2,
#                                                    no_data_val=1000.0)
#         np.testing.assert_array_almost_equal(averaged_data,
#                                              self.expected_average_2)

if __name__ == '__main__':
    unittest.main()
