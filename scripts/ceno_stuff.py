from pathlib import Path
import numpy as np
import geopandas as gpd


# from sklearn.neighbors import BallTree
#
data_dir = Path("/home/sudipta/Documents/nci/Ceno_test")
# shapefile = gpd.read_file(data_dir.joinpath("ceno_polygon_stats.shp"))
# polygons = shapefile["geometry"]
# print(polygons)
#
# index = BallTree(polygons)


# import numpy as np
# from sklearn.neighbors import KDTree, BallTree
#
# polygons = np.array([
#     [0, 0],
#     [1, 0],
#     [1, 1],
#     [0, 1]
# ])
#
# query_point = np.array([0.5, 0.5])
#
# kdtree = KDTree(polygons)
# distances, indices = kdtree.query(query_point, k=3)
#
# print(distances)
# print(indices)

import rasterio as rio

smoothed = "spline_p1_sm.tif"
ceno = "euc_geo.tif"

# import IPython; IPython.embed(); import sys; sys.exit()
src = rio.open(data_dir.joinpath(smoothed))
X = rio.open(data_dir.joinpath(smoothed)).read(masked=True)
Y = rio.open(data_dir.joinpath(ceno)).read(masked=True)
# W = rio.open(data_dir.joinpath(ceno)).read(masked=True)

# _, rows, cols = X.shape

mask = X.mask

greater = Y > 40000
less = Y < 20000
W = (40000 - Y)/20000
W[greater] = 0
W[less] = 1
W = np.ma.MaskedArray(data=W, mask=X.mask)

output = X * (1-W**2) + Y*W**2

profile = src.profile
profile.update({
    'driver': 'COG'
})

with rio.open(f'weighted_average_quadratic.tif', 'w', ** profile, compress='lzw') as dst:
    dst.write(output)


# output = X * W**2 + Y*(1-W**2)
