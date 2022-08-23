from uncoverml import shapley


tiff_loc = '/g/data/ge3/data/covariates/dem_fill.tif'
shapefile_loc = 'Shapley_ROI/Points_Shapley.shp'


if __name__ == '__main__':
    intersected_array = shapley.intersect_poly_shp(shapefile_loc, tiff_loc)
    print(type(intersected_array))
    print(intersected_array.shape)
