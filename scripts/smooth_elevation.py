from pathlib import Path
import numpy as np
from scipy.interpolate import interp2d, RegularGridInterpolator
import rasterio as rio
from rasterio.io import DatasetReader
smooth_dem =  Path("/home/sudipta/Documents/nci/smooth_dem")
elevation = smooth_dem.joinpath("elevation_small.tif")
ceno = smooth_dem.joinpath("ceno_small.tif")
relief = smooth_dem.joinpath("relief_small.tif")

with rio.open(elevation) as geotif:
    profile: rio.profiles.Profile = geotif.profile
    ele: DatasetReader = geotif.read(1, masked=True)

with rio.open(relief) as geotif:
    rel: DatasetReader = geotif.read(1, masked=True)

with rio.open(ceno) as geotif:
    cen: DatasetReader = geotif.read(1, masked=True)

ele_mod = np.ma.array(data=ele.data, mask=(ele.mask | ((cen.data > 0.01) & (rel.data > 2.3))), fill_value=np.nan)
ele_mod.data[ele_mod.mask] = np.nan
# ele_mod.mask = ele.mask | ((cen.data > 0.01) & (rel.data > 0.01))


def points_removed_interp():
    z = ele_mod
    x, y = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    x1 = x[~z.mask]
    y1 = y[~z.mask]
    z1 = z[~z.mask]
    print(z1.shape)
    return interp2d(x1, y1, z1, kind="linear")(np.arange(z.shape[0]), np.arange(z.shape[1]))


def recatangular_interp():
    z = ele_mod.filled(np.nan)
    x = np.array(range(ele_mod.shape[0]))
    y = np.array(range(ele_mod.shape[1]))
    zinterp = RegularGridInterpolator((x, y), z, method="linear")
    X2, Y2 = np.meshgrid(x, y)
    newpoints = np.array((X2, Y2)).T

    # actual interpolation
    z2 = zinterp(newpoints)
    z2_masked = np.ma.array(z2, mask=np.isnan(z2))
    return z2_masked


profile.update(compress='lzw')

with rio.open(smooth_dem.joinpath("smooth_elevation_small_masked_ref_2p3.tif"), mode="w", **profile,) as update_dataset:
    # out = recatangular_interp()
    # out = points_removed_interp()
    print(profile)
    update_dataset.write(ele_mod, 1)


# import IPython; IPython.embed(); import sys; sys.exit()
# flat_ele = interp2d(x1, y1, z1)
#    (np.arange(ele_mod.shape[0]), np.arange(ele_mod.shape[1]))



