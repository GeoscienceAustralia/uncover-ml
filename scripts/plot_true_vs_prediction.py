from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# df = pd.read_csv('~/Documents/nci/gradient_boost_optimise_oos_validation.csv')
df = pd.read_csv('~/Documents/nci/gbquantiles_oos_validation.csv')
df.columns = [c.strip() for c in df.columns]



coords = df.loc[:, ['lon', 'lat']]

dbscan = DBSCAN(eps=5000, n_jobs=-1, min_samples=10)
dbscan.fit(coords)
line_no = dbscan.labels_.astype(np.uint16)


def extent_of_data(data: pd.DataFrame) -> Tuple[float, float, float, float]:
    x_min, x_max = min(data['lat']), max(data['lon'])
    y_min, y_max = min(data['lat']), max(data['lon'])
    return x_max, x_min, y_max, y_min


def determine_and_sort_by_dominant_line_direction(line_data):
    x_max, x_min, y_max, y_min = extent_of_data(line_data)
    if abs(x_max-x_min) > abs(y_max-y_min):
        sort_by_col = 'lat'
    else:
        sort_by_col = 'lon'
    line_data = line_data.sort_values(by=[sort_by_col], ascending=[False])
    return line_data


def add_delta(line: pd.DataFrame, origin=None):
    """
    :param line:
    :param origin: origin of flight line, if not provided assumed ot be the at the lowest y value
    :return:
    """
    line = determine_and_sort_by_dominant_line_direction(line)
    line_cols = list(line.columns)
    line['POINT_X_diff'] = line['lat'].diff()
    line['POINT_Y_diff'] = line['lon'].diff()
    line['delta'] = np.sqrt(line.POINT_X_diff ** 2 + line.POINT_Y_diff ** 2)
    line['delta'] = line['delta'].fillna(value=0.0)
    if origin is not None:
        line['delta'].iat[0] = np.sqrt(
            (line.POINT_X.iat[0] - origin[0]) ** 2 +
            (line.POINT_Y.iat[0] - origin[1]) ** 2
        )
    line['d'] = line['delta'].cumsum()
    line = line.sort_values(by=['d'], ascending=True)  # must sort by distance from origin of flight line
    return line[line_cols + ['d']]

df['group'] = line_no
df = add_delta(df)


# identify which group you want to plot
def identify_group(df):
    # dfff = df[df.y_true > 0]
    plt.scatter(dfff.lat, dfff.lon)
    for txt in np.unique(dfff.group):
        dff = dfff[dfff.group == txt]
        for lat, lon in zip(dff.lat, dff.lon):
            plt.annotate(txt, (lat, lon))
    plt.show()


def plot_quantiles(line):
    xx = line.d
    y_test = line.Prediction
    y_lower = line['Lower quantile']
    y_upper = line['Upper quantile']
    plt.plot(xx, line.y_true, "b.", markersize=10, label="log10(conductivity)")
    plt.plot(xx, y_test, color='orange', linestyle='-',
             # marker='o',
             label="Predicted median")
    plt.plot(xx, y_upper, "k-")
    plt.plot(xx, y_lower, "k-")
    plt.fill_between(
        # (95362.213934-line.d).ravel()
        (line.d).ravel(),
        y_lower, y_upper, alpha=0.4, label='Predicted 90% interval'
    )
    plt.legend(loc='upper left')
    plt.ylabel('log10(conductivity)')
    plt.xlabel('Distance along flight line (meters)')


import IPython; IPython.embed(); import sys; sys.exit()
dff = df[df['group'] == 18]
plt.figure(figsize=(10, 4))
plt.semilogy(10**dff['y_true'], 'o-', color='black', label='true')
plt.semilogy(10**dff['Prediction'], 'o-', color='orange', label='prediction')
plt.legend()
plt.show()




xx = line.d
y_test = line.Prediction
y_lower = line['Lower_quan']
y_upper = line['Upper_quan']
plt.plot(xx, line.y_true, "b.", markersize=10, label="Observed log10(conductivity)")
plt.plot(xx, line.out, "g", markersize=10, linestyle='-',
         label="Kriged log10(conductivity)")
plt.plot(xx, y_test, color='orange', linestyle='-',
         # marker='o',
         label="Predicted median")
plt.plot(xx, y_upper, "k-")
plt.plot(xx, y_lower, "k-")
plt.fill_between(
    # (95362.213934-line.d).ravel()
    (line.d).ravel(),
    y_lower, y_upper, alpha=0.4, label='Predicted 90% interval'
)
plt.legend(loc='upper left')
plt.ylabel('log10(conductivity)')
plt.xlabel('Distance along flight line (meters)')
