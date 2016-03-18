import rasterio
import numpy as np

from threading import Thread
from queue import Queue, Empty

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


# Adapted from gist: https://gist.github.com/EyalAr/7915597
class NonBlockingStreamReader:

    def __init__(self, stream):
        """
        A Non-blocking stream reader class for capturing stdout/stderr streams.

        Parameters
        ----------
        stream: stream
            the stream to read from. Usually a process' stdout or stderr.
        """

        self._s = stream
        self._q = Queue()

        def _populateQueue(stream, queue):
            """
            Collect lines from 'stream' and put them in 'queque'.
            """

            while True:
                line = stream.readline()
                if line:
                    queue.put(line)
                else:
                    break  # End of stream
                    # raise UnexpectedEndOfStream

        self._t = Thread(target = _populateQueue,
                args = (self._s, self._q))
        self._t.daemon = True
        self._t.start()  # start collecting lines from the stream

    def readline(self, timeout=None):
        """
        Read a line from the stream, without blocking.

        Parameters
        ----------
        timeout: float, optional
            time in seconds to wait for stream to be populated before
            reading. None for read immediately.

        Returns
        -------
        bytes or str:
            depending on the underlying stream. If this times out, None is
            returned.
        """
        try:
            return self._q.get(block=timeout is not None, timeout=timeout)
        except Empty:
            return None

class UnexpectedEndOfStream(Exception): pass
