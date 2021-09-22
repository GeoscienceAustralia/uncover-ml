import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

df = pd.read_csv('~/Documents/nci/gradient_boost_optimise_oos_validation.csv')
# df.columns = [c.strip() for c in df.columns]

coords = df.loc[:, ['lon', 'lat']]

dbscan = DBSCAN(eps=5000, n_jobs=-1, min_samples=10)
dbscan.fit(coords)
line_no = dbscan.labels_.astype(np.uint16)

df['group'] = line_no
dff = df[df['group']==61]
plt.figure(figsize=(10, 4))
plt.semilogy(10**dff['y_true'], 'o-', color='black', label='true')
plt.semilogy(10**dff['Prediction'], 'o-', color='orange', label='prediction')
plt.legend()
plt.show()
