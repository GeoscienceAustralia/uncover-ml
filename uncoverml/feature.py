import os.path
import numpy as np
import rasterio
import tables as hdf
from uncoverml.celerybase import celery
from uncoverml import patch


def load_window(geotiff, img_slices):
    x_slice, y_slice = img_slices
    window = ((x_slice.start, x_slice.stop), (y_slice.start, y_slice.stop))
    with rasterio.open(geotiff) as raster:
        w = raster.read(window=window)
        w = w[np.newaxis,:,:] if w.ndim == 2 else w
        w = np.rollaxis(w, 0, 3) #channels at the back!
    return w

def output_window(patches, chunk, outfile):
    filename = os.path.splitext(outfile)[0] + "_" + str(chunk) + ".hdf5"
    h5file = hdf.open_file(filename, mode='w')
    array_shape = patches.shape

    filters = hdf.Filters(complevel=5, complib='zlib')
    h5file.create_carray("/", "features".format(chunk), filters=filters,
                         atom=hdf.Float64Atom(), shape=array_shape)
    h5file.root.features[:] = patches
    h5file.close()

def transform(x):
    return x.flatten()

@celery.task(name='process_window')
def process_window(geotiff, chunk, img_slice, pointspec, patchsize, 
                   transform, outfile):
    img = load_window(geotiff, img_slice)
    patches = patch.patches(img, pointspec, patchsize)
    processed_patches = map(transform, patches)
    data = np.array(list(processed_patches), dtype=float)
    output_window(data, chunk, outfile)

