#! /usr/bin/env python3
import logging
import sys
import click
import os.path
import numpy as np
import rasterio
import tables as hdf
from affine import Affine

log = logging.getLogger(__name__)



@click.command()
@click.option('--output', type=click.Path(exists=True), required=1)
@click.option('--verbose', help="Log everything", default=False)
@click.argument('geotiffs', nargs=-1)
def main(output, verbose, geotiffs):
    """ Add one or more geotiffs to an hdf5 file stack (that may exist)

        The HDF5 file has the following datasets:

            - Raster: (original image data)

            - Latitude: (vector or matrix of pixel latitudes)

            - Longitude: (vector or matrix of pixel longitudes)

            - Label: (label or descriptions of bands)

    """

    # setup logging
    if verbose is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    hdf_exists = os.path.isfile(output)

    # If hdf5 does NOT exist:
    #   With the first geotiff in input:
    #       transform lat lons to pixel centres
    #       Add the first geotiff to a new hdf5 file
    #       remove first geotiff from input list
    #
    # For each geotiff in input:
    #   transform lat lons to pixel centres
    #   Check geotiff is consistent with hdf5
    #
    # For each geotiff in input:
    #   Add geotiff as new layer to hdf
    #
    # exit

    if hdf_exists:
        try:
            h5file = hdf.open_file(output, mode="r")
        except:
            log.critical("could not open output file")
            sys.exit(-1)

        # import IPython; IPython.embed()

        longitude = h5file.get_node("/Longitude")
        latitude = h5file.get_node("/Latitude")

        xres = len(longitude)
        yres = len(latitude)
        xbounds = (longitude[0], longitude[-1])
        ybounds = (latitude[0], latitude[-1])

        # validate every input
        for filename in geotiffs:
            with rasterio.open(filename) as f:

                if not f.width == xres:
                    log.critical("input image width does not match hdf5")
                    sys.exit(-1)
                if not f.height == yres:
                    log.critical("input image height does not match hdf5")
                    sys.exit(-1)

                f_lats, f_lons = align_latlon_pix(f)

                if not xbounds == (f_lons[0], f_lons[-1]):
                    log.critical("image x-bounds do not match hdf5")
                    sys.exit(-1)
                if not ybounds == (f_lats[0], f_lats[-1]):
                    log.critical("image y-bounds do not match hdf5")
                    sys.exit(-1)
                if not np.all(longitude, f_lons):
                    log.critical("longitude pixel values do not match hdf5")
                    sys.exit(-1)
                if not np.all(latitude, f_lats):
                    log.critical("latitude pixel values do not match hdf5")
                    sys.exit(-1)

                # get xres, yres, xbounds, ybounds
    else:
        h5file = hdf.open_file(output, mode='w')

    # 

    # I = f.read()
    # nanvals = f.get_nodatavals()

    # # Permute layers to be more like a standard image, i.e. (band, lon, lat) ->
    # #   (lon, lat, band)
    # I = (I.transpose([2, 1, 0]))[:, ::-1]
    # lats = lats[::-1]

    # # build channel labels
    # basename = os.path.basename(raster).split(".")[-2]
    # print("basename: " + basename)
    # channel_labels = np.array([basename + "_band_" + str(i+1) 
    #         for i in range(I.shape[2])], dtype='S')

    # # Mask out NaN vals if they exist
    # if nanvals is not None:
    #     for v in nanvals:
    #         if v is not None:
    #             if verbose:
    #                 print("Writing missing values")
    #             I[I == v] = np.nan

    # # Now write the hdf5
    # if verbose:
    #     print("Writing HDF5 file ...")

    # file_stump = os.path.basename(raster).split('.')[-2]
    # hdf5name = os.path.join(outputdir, file_stump + ".hdf5")
    # with h5py.File(hdf5name, 'w') as f:
    #     drast = f.create_dataset("Raster", I.shape, dtype=I.dtype, data=I)
    #     drast.attrs['affine'] = T1
    #     for k, v in crs.items():
    #         drast.attrs['k'] = v
    #     f.create_dataset("Latitude", lats.shape, dtype=float, data=lats)
    #     f.create_dataset("Longitude", lons.shape, dtype=float, data=lons)
    #     f.create_dataset("Labels", data=channel_labels)

    # if verbose:
    #     print("Done!")


# def read_raster(filename):
#     with rasterio.open(filename) as f:
#         data = rasterio_file.read()
#         nanvals = rasterio_file.get_nodatavals()
#         # must be 3d
#         assert(data.ndim == 3)
#         return data, nanvals


def align_latlon_pix(f):

    # Get affine transform for pixel centres
    # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
    T1 = f.affine * Affine.translation(0.5, 0.5)

    # No shearing or rotation allowed!!
    if not ((T1[1] == 0) and (T1[3] == 0)):
        log.critical("transform to pixel coordinates has rotation "
                     "or shear")
        sys.exit(-1)

    # compute the tiffs lat/lons
    f_lons = T1[2] + np.arange(f.width) * T1[0]
    f_lats = T1[5] + np.arange(f.height) * T1[4]
    return f_lons, f_lats


if __name__ == "__main__":
    main()
