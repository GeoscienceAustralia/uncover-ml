# /g/data/ge3/sudipta/jobs/landshark_models/lake_eyre_cropped/04_11_2022/hw1_mean_318_5folds_complicated2/mean_318_std.tif
# /g/data/ge3/sudipta/jobs/landshark_models/lake_eyre_cropped/04_11_2022/hw1_mean_318_5folds_complicated2/mean_318.tif


import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio


# with rasterio.open(tif) as src:
#     # resample data to target shape
#     data = src.read(
#         out_shape=(
#             src.count,
#             int(src.height / downscale_factor),
#             int(src.width / downscale_factor)
#         ),
#         resampling=rasterio.enums.Resampling.bilinear
#     )
#     # scale image transform
#     transform = src.transform * src.transform.scale(
#         (src.width / data.shape[-1]),
#         (src.height / data.shape[-2])
#     )
#     pts["rows"], pts["cols"] = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])


def raster_points() -> gpd.GeoDataFrame:
    with rio.Env():
        with rio.open(
                '/g/data/ge3/sudipta/jobs/landshark_models/lake_eyre_cropped/04_11_2022/hw1_mean_318_5folds_complicated2/mean_318.tif') as src:
            crs = src.crs
            # create 1D coordinate arrays (coordinates of the pixel center)
            xmin, ymax = np.around(src.xy(0.00, 0.00), 9)  # src.xy(0, 0)
            xmax, ymin = np.around(src.xy(src.height - 1, src.width - 1), 9)  # src.xy(src.width-1, src.height-1)
            x = np.linspace(xmin, xmax, src.width)
            y = np.linspace(ymax, ymin, src.height)  # max -> min so coords are top -> bottom

            # create 2D arrays
            xs, ys = np.meshgrid(x, y)
            zs = src.read(1)

            # Apply NoData mask
            mask = src.read_masks(1) > 0
            xs, ys, zs = xs[mask], ys[mask], zs[mask]
    data = {"X": pd.Series(xs.ravel()),
            "Y": pd.Series(ys.ravel()),
            "Z": pd.Series(zs.ravel())}
    df = pd.DataFrame(data=data)
    geometry = gpd.points_from_xy(df.X, df.Y)
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    return gdf


raster_gdf = raster_points()

shape_file = ""
shp_gdf = gpd.read_file(shape_file)
gpd.overlay(raster_gdf, shp_gdf, how="intersection")
print(raster_gdf.head())

