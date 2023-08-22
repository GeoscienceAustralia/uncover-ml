import numpy as np
import rasterio as rio
near_rough = "/g/data/jl14/new_ceno_inputs/KD_8000_80m.tif"
far_smooth = "/g/data/jl14/new_ceno_inputs/KD_3500_80m.tif"
scale = "/g/data/jl14/80m_covarites/proximity/P_Dist_Meso.tif"

src = rio.open(far_smooth)
far = rio.open(far_smooth).read(masked=True)
near = rio.open(near_rough).read(masked=True)
Z = rio.open(scale).read(masked=True)
mask = far.mask

greater = Z > 0.43
less = Z < 0.05
W = (0.43 - Z)/(0.43 - 0.05)
W[greater] = 0
W[less] = 1
W = np.ma.MaskedArray(data=W, mask=far.mask)

output = far * (1 - W ** 2) + near * W ** 2

profile = src.profile
profile.update({'driver': 'COG', 'BIGTIFF': 'YES'})

with rio.open(f'kd_8000_3500_dist_meso_weighted_average_quadratic_bigtiff.tif', 'w', ** profile, compress='lzw') as dst:
    dst.write(output)



import numpy as np
import rasterio as rio
near_rough = "/g/data/jl14/80m_covarites/terrain/T_DEM_S.tif"
far_smooth = "/g/data/jl14/new_ceno_inputs/80m_DEM_smooth_new.tif"
scale = "/g/data/jl14/80m_covarites/proximity/P_Dist_Meso.tif"

src = rio.open(far_smooth)
far = rio.open(far_smooth).read(masked=True)
near = rio.open(near_rough).read(masked=True)
Z = rio.open(scale).read(masked=True)
mask = far.mask

lower_end = 0.01
higher_end = 0.08

greater = Z > higher_end
less = Z < lower_end
W = (higher_end - Z)/(higher_end - lower_end)
W[greater] = 0
W[less] = 1
W = np.ma.MaskedArray(data=W, mask=far.mask)

output = far * (1 - W ** 2) + near * W ** 2

profile = src.profile
profile.update({'driver': 'COG', 'BIGTIFF': 'YES'})

with rio.open(f'dem_s_DEM_smooth_new_dist_meso_weighted_average_quadratic_bigtiff.tif', 'w', ** profile, compress='lzw') as dst:
    dst.write(output)


# for the DEM use /g/data/jl14/80m_covarites/proximity/P_Dist_Meso.tif to determing the weights and transition to combine the two DEM grids.
# Use the values 0.01 to 0.15 i.e. < 0.01 100% grid 1 and > 0.15 100% of grid 2
# grid 1 == /g/data/jl14/80m_covarites/terrain/T_DEM_S.tif and grid 2 == /g/data/jl14/new_ceno_inputs/Spline_DEM_80.tif