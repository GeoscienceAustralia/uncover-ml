"""
Currently only supports single band raster.
"""
from optparse import OptionParser
import numpy as np
from osgeo import gdal, gdalconst
from skimage.exposure import equalize_adapthist, rescale_intensity


def apply_clahe(input_raster, output_raster,
                in_range="image", kernel_size=200, clip_limit=0.1):
    # don't change the next two lines

    if (in_range is not None) and (in_range != "image"):
        in_range = tuple([float(s) for s in in_range.split()])
    else:
        in_range = 'image'

    src = gdal.Open(input_raster, gdalconst.GA_ReadOnly)
    data = src.GetRasterBand(1).ReadAsArray()
    nodata = getattr(np, str(data.dtype))(
        src.GetRasterBand(1).GetNoDataValue())
    mask = data == nodata
    output_nodata = src.GetRasterBand(1).GetNoDataValue()

    datatype = src.GetRasterBand(1).DataType
    # import IPython; IPython.embed()

    # rescale the data as the adapthist expects data in (-1, 1)
    # specify in_range=(min_value_to_limit, max_value_to_limit),
    # in_range="image" to use min and max in the image values.
    rescaled_data = rescale_intensity(data.astype(np.float32),
                                      in_range=in_range,
                                      out_range=(0, 1))

    # kernel_size: integer or list-like, optional
    # Defines the shape of contextual regions used in the algorithm.
    # If iterable is passed, it must have the same number of elements
    # as image.ndim (without color channel).
    # If integer, it is broadcasted to each image dimension. By default,
    # kernel_size is 1/8 of image height by 1/8 of its width.
    #
    # clip_limit : float, optional
    # Clipping limit, normalized between 0 and 1 (higher values give more contrast).
    #
    # nbins : int, optional
    # Number of gray bins for histogram (“data range”).

    stretched_data = equalize_adapthist(rescaled_data,
                                        kernel_size=kernel_size,
                                        clip_limit=clip_limit,
                                        nbins=256)
    # don't change anything below this
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_raster,
                           xsize=src.RasterXSize,
                           ysize=src.RasterYSize,
                           bands=1,
                           eType=datatype
                           )
    out_ds.SetGeoTransform(src.GetGeoTransform())
    out_ds.SetProjection(src.GetProjection())
    rescaled_data[mask] = output_nodata
    out_ds.GetRasterBand(1).WriteArray(stretched_data)
    # might have to change dtype manually
    out_ds.GetRasterBand(1).SetNoDataValue(output_nodata)
    out_ds.FlushCache()


if __name__ == '__main__':
    parser = OptionParser(usage='%prog -i input_raster -o output_raster \n'
                                '-r input_range -k kernel_size \n'
                                '-c clip_limit')
    parser.add_option('-i', '--input_raster', type=str, dest='input_raster',
                      help='name of input raster file')

    parser.add_option('-o', '--output_raster', type=str, dest='output_raster',
                      help='name of output raster file')

    parser.add_option('-r', '--input_range', type=str, dest='input_range',
                      help="input range of the input raster to use."
                           " Can use tuple like '125 3000'")

    parser.add_option('-k', '--kernel_size', type=int, dest='kernel_size',
                      default=400,
                      help='kernel size to use')

    parser.add_option('-c', '--clip_limit', type=float, dest='clip_limit',
                      default='0.01',
                      help='clip limit used in clahe')

    options, args = parser.parse_args()

    if not options.input_raster:  # if filename is not given
        parser.error('Input raster filename must be provided.')

    if not options.output_raster:  # if filename is not given
        parser.error('Output raster filename must be provided.')

    apply_clahe(options.input_raster, options.output_raster,
                options.input_range,  options.kernel_size, options.clip_limit)