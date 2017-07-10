import os
import shutil
from os.path import join, basename, exists

from osgeo import gdal
from preprocessing import crop_mask_resample_reproject as crop
from preprocessing.crop_mask_resample_reproject import Options

# The mock files have extents = '900000 -4300000 940000 -4260000'
mock_extents = [920000, -4300000, 930000, -4280000]

def test_geotransform_projection_nodata(mock_files, random_filename):
    tmp_output = random_filename(ext='.tif')
    extents = [str(s) for s in [920000, -4300000, 929000, -4290000]]

    # the mock is already geotransformed, so this will have no effect
    # to projection and nodata, but will be cropped making the
    # geotransform tuple different
    crop.crop_reproject_resample(mock_files['std2000_no_mask'], tmp_output,
                                 sampling='bilinear',
                                 extents=extents,
                                 reproject=True)

    ds = gdal.Open(tmp_output)
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None

    ds = gdal.Open(mock_files['std2000_no_mask'])
    projection_input = ds.GetProjection()
    nodata_input = ds.GetRasterBand(1).GetNoDataValue()
    ds = None

    assert nodata == nodata_input
    assert projection == projection_input
    assert gt[1] == float(crop.OUTPUT_RES[0])
    assert gt[0] == float(extents[0])
    assert gt[3] == float(extents[3])
    os.remove(tmp_output)

def test_apply_mask(mock_files, random_filename):
    output_file = random_filename(ext='.tif')
    jpeg = False
    tmp_out_file = random_filename(ext='.tif')
    shutil.copy(mock_files['std2000_no_mask'], tmp_out_file)
    crop.apply_mask(mask_file=mock_files['mask'],
                    tmp_output_file=tmp_out_file,
                    output_file=output_file,
                    jpeg=jpeg)

    ds = gdal.Open(output_file)
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None

    ds = gdal.Open(mock_files['std2000'])
    projection_input = ds.GetProjection()
    nodata_input = ds.GetRasterBand(1).GetNoDataValue()
    ds = None

    assert nodata == nodata_input
    assert projection == projection_input
    assert gt[1] == float(crop.OUTPUT_RES[0])
    assert gt[0] == mock_extents[0]
    assert gt[3] == mock_extents[3]
    os.remove(output_file)

def test_do_work(mock_files, random_filename):
    # input_file, mask_file, output_file, resampling, extents, jpeg
    output_file = random_filename(ext='.tif')
    options = Options(resampling='bilinear',
                      extents=mock_extents,
                      jpeg=True,
                      reproject=True)
    crop.do_work(input_file=mock_files['std2000_no_mask'],
                 output_file=output_file,
                 options=options,
                 mask_file=mock_files['mask'])

    # output file was created
    assert exists(output_file)

    # assert jpeg was created
    assert exists(output_file.rsplit('.')[0] + '.jpg')

    os.remove(output_file)
