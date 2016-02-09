import logging
import sys
import click
import os.path
import numpy as np
import rasterio
import click
import tables as hdf
from affine import Affine

log = logging.getLogger(__name__)


def read_raster(filename):
    with rasterio.open(filename) as f:
        data = rasterio_file.read()
        nanvals = rasterio_file.get_nodatavals()
        # must be 3d
        assert(data.ndim == 3)
        return data, nanvals


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
    if hdf_exists:
        try:
            h5file = open_file(output, mode="r")
        except:
            log.critical("could not open output file")
            sys.exit(-1)

        longitude = h5file.get_node("Longitude")
        latitude = h5file.get_node("Latitude")

        xres = longitude.shape()
        yres = latitude.shape()
        xbounds = (longitude[0], longitude[-1])
        ybounds = (latitude[0], latitude[-1])
        # validate every input
        for filename in geotiffs:
            with rasterio.open(filename) as f:
                # Get affine transform for pixel centres
                T1 = f.affine * Affine.translation(0.5, 0.5)
                # No shearing or rotation allowed!!
                if not ((T1[1] == 0) and (T1[3] == 0)):
                    log.critical("transform to pixel coordinates has rotation "
                    "or shear")
                    sys.exit(-1)
                # compute the tiffs lat/lons
                f_lons = T1[2] + np.arange(f.width) * T1[0]
                f_lats = T1[5] + np.arange(f.height) * T1[4]
                if not f.width == xres:
                    log.critical("input image width does not match hdf5")
                    sys.exit(-1)
                if not f.height == yres:
                    log.critical("input image height does not match hdf5")
                    sys.exit(-1)
                if not xbounds == (f_lons[0],f_lons[-1]):
                    log.critical("image x-bounds do not match hdf5")
                    sys.exit(-1)
                if not ybounds == (f_lats[0], f_lats[-1]):
                    log.critical("image y-bounds do not match hdf5")
                    sys.exit(-1)
                if not np.all(longitude,f_lons):
                    log.critical("longitude pixel values do not match hdf5")
                    sys.exit(-1)
                if not np.all(latitude, f_lats):
                    log.critical("latitude pixel values do not match hdf5")
                    sys.exit(-1)

                #now we know we're all good
                f.width == xres and \
                f.height == yres and 

            # Just find lat/lons of axis if there is no rotation/shearing
            # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
                import IPython; IPython.embed()
                # get xres, yres, xbounds, ybounds
    else:
        h5file = open_file(output, mode='w')


        



        # I = f.read()
        # nanvals = f.get_nodatavals()

if __name__ == "__main__":
    main()
