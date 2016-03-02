import numpy as np

cube = np.load('cube.npy')
label_names = np.load('label_names.npy') 
label_coords = np.load('label_coords.npy')
labels = np.load('labels.npy')
lats = np.load('lats.npy')
lons = np.load('lons.npy')
x_bands = np.load('x_bands.npy')

pixsize_x = (lons[-1] - lons[0])/float(lons.shape[0])
pixsize_y = (lats[-1] - lats[0])/float(lats.shape[0])

nlabels = label_coords.shape[0]
label_idx = np.zeros((nlabels,2), dtype=int)
for i in range(nlabels):
    xidx = int((label_coords[i,0] - lons[0])/float(pixsize_x) + 0.5)
    yidx = int((label_coords[i,1] - lats[0])/float(pixsize_y) + 0.5)
    assert (xidx >= 0 and xidx < lons.shape[0])
    assert (yidx >= 0 and yidx < lats.shape[0])
    label_idx[i, 0] = xidx
    label_idx[i, 1] = yidx

ndims = cube.shape[2]
features = np.zeros((nlabels, ndims), dtype=float)
for i in range(nlabels):
    features[i] = cube[label_idx[i,0], label_idx[i,1]]


train_X = features
train_y = labels
test_X = cube.reshape((-1,ndims))

np.save('X.npy', train_X)
np.save('y.npy', train_y)
np.save('Xstar.npy', test_X)
