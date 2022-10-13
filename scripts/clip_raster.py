import fiona
import rasterio
import rasterio.mask


def clip_raster(shapefile, in_rast, out_name):
    with fiona.open(shapefile, 'r') as shp:
        shapes = [feature["geometry"] for feature in shp]

    with rasterio.open(in_rast) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

    out_file = out_name + '.tif'
    with rasterio.open(out_file, 'w', **out_meta) as dest:
        dest.write(out_image)


if __name__ == '__main__':
    shape_file = '/g/data/ge3/as6887/projects/uncoverml_models/cluster-test-small/extent/K-means_extent.shp'
    file_list = [
        ('/g/data/ge3/sudipta/80m_albers/climate/Clim_Prescott_LindaGregory.tif', 'climate'),
        ('/g/data/ge3/sudipta/80m_albers/terrain/terrain_ruggedness_index_dia_320m.tif', 'ruggedness'),
        ('/g/data/ge3/sudipta/80m_albers/models/WeatheringIntensityIndex.tif', 'weathering'),
        ('/g/data/ge3/sudipta/80m_albers/maps/mask_80m_albers_cogs.tif', 'mask')
    ]
    for rast in file_list:
        print(rast[1])
        clip_raster(shape_file, rast[0], rast[1])
