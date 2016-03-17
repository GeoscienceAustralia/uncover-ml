import click
import rasterio
import tables
import geom

@cl.command()
@cl.option('--pointspec', type=cl.Path(exists=True), required=True)
@cl.option('--verbose', help="Log everything", default=False)
@cl.option('--outfile',type=cl.Path(exists=False),required=False,
           help="Optional name of output file")
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
def buildInput(geotiff, pointspec, verbose, outfile=None):
    """Extract the pixels of the geotiff according to pointspec and
    output to HDF5
    """
    
    #generate output filename if not existing
    if outfile is None:
        outfile = os.path.splitext(geotiff)[0] + ".hdf5"

    # See this link for windowed reads:
    #https://github.com/mapbox/rasterio/blob/master/docs/windowed-rw.rst
    with rasterio.open(geotiff) as raster:
        longitudes, latitudes = geom.lonlat_pixel_centres(raster)
        ndims = raster.count
    
    h5file = hdf.open_file(output, mode='w')

    # Now that we know the images are all the same size/shape, we use the first
    # one to calculate the appropriate latitudes/longitudes
    with rasterio.open(geotiffs[0]) as raster:
        longs, lats = lonlat_pixel_centres(raster)
        h5file.create_array("/", "Longitude", obj=longs)
        h5file.create_array("/", "Latitude", obj=lats)
        raster_shape = (longs.shape[0], lats.shape[0], ndims)
        h5file.create_carray("/", "Raster", atom=hdf.Float64Atom(),
                shape=raster_shape)
        h5file.create_array("/", "Labels", atom=hdf.StringAtom(itemsize=100),
                shape=(ndims,))

    #Write the actual data into the hdf5 file
    # note we keep note of the current band index we're writing into
    idx = 0
    for f in geotiffs:
        idx = write_to_hdf5(f, h5file, idx)


def write_to_hdf5(raster, h5file, idx):
    log.info("Writing {} into hdf5 array".format(raster))
    with rasterio.open(raster) as f:
        I = f.read()
        nanvals = f.get_nodatavals()
        ndims = f.count

    # Permute layers to be less like a standard image and more like a
    # matrix i.e. (band, lon, lat) -> (lon, lat, band)
    I = (I.transpose([2, 1, 0]))[:, ::-1]

    # build channel labels
    basename = os.path.basename(raster).split(".")[-2]
    print("basename: " + basename)
    channel_labels = np.array([basename + "_band_" + str(i+1) 
            for i in range(I.shape[2])], dtype='S')

    # Mask out NaN vals if they exist
    if nanvals is not None:
        for v in nanvals:
            if v is not None:
                I[I == v] = np.nan

    # Now write the hdf5
    h5file.root.Raster[:,:,idx:idx+ndims] = I
    h5file.root.Labels[idx:idx+ndims] = channel_labels
    return idx + ndims

