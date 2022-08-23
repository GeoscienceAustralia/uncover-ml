from uncoverml import shapley


tiff_loc = '/g/data/ge3/data/covariates/dem_fill.tif'
shapefile_loc = '/g/data/ge3/as6887/projects/uncoverml_models/median_0_50/'


if __name__ == '__main__':
    intersected_array = shapley.intersect_poly_shp(shapefile_loc, tiff_loc)
    print(type(intersected_array))
    print(intersected_array.shape)
