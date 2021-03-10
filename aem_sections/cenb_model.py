from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits import mplot3d; import matplotlib.pyplot as plt

aem_folder = '/home/sudipta/Downloads/aem_sections'
targets = pd.read_csv(Path(aem_folder).joinpath('Albers_cenozoic_interp.txt'))
line = targets[targets['line'] == 3]
line = line.sort_values(by='Y_coor', ascending=False)

line['X_coor_diff'] = line['X_coor'].diff()
line['Y_coor_diff'] = line['Y_coor'].diff()
line['delta'] = np.sqrt(line.X_coor_diff ** 2 + line.Y_coor_diff ** 2)
line['delta'] = line['delta'].fillna(value=0.0)

line['d'] = line['delta'].cumsum()
line = line.sort_values(by=['d'], ascending=True)

# bouning_box =

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(line1.X_coor, line1.Y_coor, line1.AusAEM_DEM - line1.DEPTH, 'gray')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('depth')
# plt.show()


# plt.plot(line1.d.values, line1.AusAEM_DEM - line1.DEPTH); plt.show()


# verification y's
y = line.AusAEM_DEM - line.DEPTH

data = pd.read_csv(Path(aem_folder).joinpath('Albers_data_AEM_SB.csv'))

import IPython; IPython.embed(); import sys; sys.exit()
