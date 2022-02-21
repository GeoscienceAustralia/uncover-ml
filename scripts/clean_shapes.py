import numpy as np
import geopandas as gpd
from .intersect_rasters import geotifs

df2 = gpd.GeoDataFrame.from_file('/g/data/ge3/john/MAJORS/Nat_Fe_albers.shp')
df3 = df2[list(geotifs.values())]
# df4 = df3.loc[df3.isna().sum(axis=1) ==0, :]
df4 = df2.loc[(df3.isna().sum(axis=1) == 0) & ((np.abs(df3) < 1e10).sum(axis=1) == 31), :]
df5 = df2.loc[~((df3.isna().sum(axis=1) == 0) & ((np.abs(df3) < 1e10).sum(axis=1) == 31)), :]
