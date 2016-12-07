import unittest
import os
from os.path import join, basename, exists
import tempfile
import shutil
from osgeo import gdal
from preprocessing import crop_mask_resample_reproject as crop
UNCOVER = os.environ['UNCOVER']  # points to the uncover-ml directory


class TestCropReSampleReProject(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mocks = join(UNCOVER, 'preprocessing', 'mocks')
        # These file shave extents = '900000 -4300000 940000 -4260000'
        cls.mask = join(mocks, 'mask.tif')
        cls.std2000 = join(mocks, 'std2000.tif')
        cls.std2000_no_mask = join(mocks, 'std2000_no_mask.tif')
        cls.extents = [920000, -4300000, 930000, -4280000]

    def test_geotransform_projection_nodata(self):
        tmp_output = tempfile.mktemp(suffix='.tif')
        extents = [str(s) for s in [920000, -4300000, 929000, -4290000]]

        # the mock is already geotransformed, so this will have no effect
        # to projection and nodata, but will be cropped making the
        # geotransform tuple different
        crop.crop_reproject_resample(self.std2000_no_mask, tmp_output,
                                     sampling='bilinear',
                                     extents=extents,
                                     reproject=True)

        ds = gdal.Open(tmp_output)
        gt = ds.GetGeoTransform()
        projection = ds.GetProjection()
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        ds = None

        ds = gdal.Open(self.std2000_no_mask)
        projection_input = ds.GetProjection()
        nodata_input = ds.GetRasterBand(1).GetNoDataValue()
        ds = None

        self.assertEqual(nodata, nodata_input)
        self.assertEqual(projection, projection_input)
        self.assertEqual(gt[1], float(crop.OUTPUT_RES[0]))
        self.assertEqual(gt[0], float(extents[0]))
        self.assertEqual(gt[3], float(extents[3]))
        os.remove(tmp_output)

    def test_apply_mask(self):
        output_file = tempfile.mktemp(suffix='.tif')
        jpeg = False
        tmp_out_file = tempfile.mktemp(suffix='.tif')
        shutil.copy(self.std2000_no_mask, tmp_out_file)
        crop.apply_mask(mask_file=self.mask,
                        tmp_output_file=tmp_out_file,
                        output_file=output_file,
                        jpeg=jpeg)

        ds = gdal.Open(output_file)
        gt = ds.GetGeoTransform()
        projection = ds.GetProjection()
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        ds = None

        ds = gdal.Open(self.std2000)
        projection_input = ds.GetProjection()
        nodata_input = ds.GetRasterBand(1).GetNoDataValue()
        ds = None

        self.assertEqual(nodata, nodata_input)
        self.assertEqual(projection, projection_input)
        self.assertEqual(gt[1], float(crop.OUTPUT_RES[0]))
        self.assertEqual(gt[0], self.extents[0])
        self.assertEqual(gt[3], self.extents[3])
        os.remove(output_file)

    def test_do_work(self):
        # input_file, mask_file, output_file, resampling, extents, jpeg
        output_file = tempfile.mktemp(suffix='.tif')
        crop.do_work(input_file=self.std2000_no_mask,
                     mask_file=self.mask,
                     output_file=output_file,
                     resampling='bilinear',
                     extents=[str(s) for s in self.extents],
                     jpeg=True,
                     reproject=True)

        # output file was created
        self.assertTrue(exists(output_file))

        # assert jpeg was created
        self.assertTrue(exists(
            join(crop.TMPDIR, basename(output_file).split('.')[0] + '.jpg')))

        os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
