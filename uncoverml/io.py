import numpy as np
import rasterio

# re-badge rasterio open so all we need is io
open_raster = rasterio.open


def read_raster(geotiff, window_slices=None):
    """
    Reads data from a geotiff out outputs in (x,y,band) format. Optionally
    takes a window for getting data in a region defined by a pair
    of slices

    Parameters
    ----------
        geotiff: rasterio raster
            the geotiff file opened by rasterio
        window_slices: tuple
            A tuple of two numpy slice objects of the form (x_slice, y_slice)
            specifying the pixel index ranges in the geotiff.

    Returns
    -------
        image: array
            a 3D numpy array of shape (size_x, size_y, nbands). The type is
            the same as the input data.

    NOTE
    ----
        x - corresponds to image COLS (Lons)
        y - corresponds to image ROWS (Lats)
    """
    if window_slices is not None:
        x_slice, y_slice = window_slices
        # tanspose the slices since we are reading the original geotiff
        window = ((y_slice.start, y_slice.stop), (x_slice.start, x_slice.stop))
    else:
        window = None

    d = geotiff.read(window=window)
    d = d[np.newaxis, :, :] if d.ndim == 2 else d
    d = np.transpose(d, [2, 1, 0])  # Transpose and channels at back
    return d

