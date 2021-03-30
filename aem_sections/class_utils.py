import logging
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from mpl_toolkits import mplot3d; import matplotlib.pyplot as plt

from aem_sections.utils import extract_required_aem_data, convert_to_xy, create_interp_data, create_train_test_set

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

LINE_NO = 1

aem_folder = '/home/sudipta/Downloads/aem_sections'
log.info("reading interp data...")
targets = pd.read_csv(Path(aem_folder).joinpath('Albers_cenozoic_interp.txt'))
line = targets[(targets['Type'] != 'WITHIN_Cenozoic') & (targets['line'] == LINE_NO)]
line = line.sort_values(by='Y_coor', ascending=False)
line['X_coor_diff'] = line['X_coor'].diff()
line['Y_coor_diff'] = line['Y_coor'].diff()
line['delta'] = np.sqrt(line.X_coor_diff ** 2 + line.Y_coor_diff ** 2)
x_line_origin, y_line_origin = line.X_coor.values[0], line.Y_coor.values[0]
line['delta'] = line['delta'].fillna(value=0.0)

line['d'] = line['delta'].cumsum()
line = line.sort_values(by=['d'], ascending=True)

x_min, x_max = min(line.X_coor.values), max(line.X_coor.values)
y_min, y_max = min(line.Y_coor.values), max(line.Y_coor.values)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(line.X_coor, line.Y_coor, line.AusAEM_DEM - line.DEPTH, 'gray')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('depth')
# plt.show()

log.info("reading covariates")
original_aem_data = pd.read_csv(Path(aem_folder).joinpath('Albers_data_AEM_SB.csv'))
all_interp_data = pd.read_csv(Path(aem_folder).joinpath('Albers_cenozoic_interp.txt'))

all_lines = create_interp_data(all_interp_data, included_lines=[LINE_NO])
aem_xy_and_other_covs, aem_conductivities, aem_thickness = extract_required_aem_data(original_aem_data, all_lines)

import pickle

if not Path('covariates_targets_class.data').exists():
    data = convert_to_xy(aem_xy_and_other_covs, aem_conductivities, aem_thickness, all_lines)
    log.info("saving data on disc for future use")
    pickle.dump(data, open('covariates_targets_class.data', 'wb'))
else:
    log.warning("Reusing data from disc!!!")
    data = pickle.load(open('covariates_targets.data', 'rb'))

X = data['covariates']
from aem_sections.utils import add_delta
X = add_delta(X)
from scipy.interpolate import SmoothBivariateSpline

# for c in X.columns:
c = 'conductivity'
interpolator = SmoothBivariateSpline(x=X['d']/1000, y=X['Z_coor'], z=X[c])

d = np.linspace(min(X.d)/1000, max(X.d)/1000, 100)
depths = np.linspace(min(X.Z_coor), max(X.Z_coor), 100)  # y
mesh = np.meshgrid(d, depths)
cov = interpolator.ev(* mesh)

fig, ax = plt.subplots()
ax.contourf(d, depths, cov)
ax.xaxis.grid(True, zorder=0)
ax.yaxis.grid(True, zorder=0)

ax.plot(line.d/1000, line.DEPTH, color="c", linewidth=2)
ax.set_xlabel('Distance')
ax.set_ylabel('Depth')


from osgeo import gdal

ds = gdal.Open('tests/test_data/sirsam/covariates/dem_foc2.tif')
driver = gdal.GetDriverByName('GTiff')
outds = driver.Create('conductivity.tif', len(d), len(depths), 1, gdal.GDT_Float32)
outds.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
outds.SetProjection(ds.GetProjection())##sets same projection as input
band = outds.GetRasterBand(1)
band.SetNoDataValue(10000)##if you want these values transparent
band.WriteArray(cov)
outds.FlushCache()  # write on the disc
band = None
outds = None
ds = None

import IPython; IPython.embed(); import sys; sys.exit()




# interpolator = SmoothBivariateSpline(x=mesh[0], y=mesh[1], z=)



import IPython; IPython.embed(); import sys; sys.exit()

plt.plot(line.d.values, line.DEPTH)
plt.show()



# 1 there are two typesof CEN-B lines

import sys; sys.exit()
