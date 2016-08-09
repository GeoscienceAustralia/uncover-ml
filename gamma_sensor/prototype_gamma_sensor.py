#!/usr/bin/env python3

""" Investigating the Gamma sensor for the GA-cover project
    Disclaimer: prototype code.
"""
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import shapefile  # package pyshp
from scipy import sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import spsolve

from tqdm import tqdm
import nvector as nv
import numpy as np
import pyproj
import percache
from bisect import bisect_left
import logging
import os

log = logging.getLogger(name=__name__)
data_root = os.path.expanduser('~/Data/GA-cover/')
cache = percache.Cache(data_root + 'cache')
mass_attenuation_air = 0.09  # assume some sort of bulk property
density_air = 1.22
mu_air = mass_attenuation_air * density_air



def main():
    """Main demo."""

    # Load survey data
    llh, data = get_flightlines()
    # llh in n*3 lon-lat-hei format
    # data in n*3 k th u format

    # Crop to ROI
    ROI = (120.4, 120.5, -27.4, -27.3)
    keep = ((llh[:, 0] > ROI[0]) *
            (llh[:, 0] < ROI[1]) *
            (llh[:, 1] > ROI[2]) *
            (llh[:, 1] < ROI[3]))
    llh = llh[keep]
    data = data[keep]

    # Change reference frame to local North/East/Up in metres
    frame = LocalFrame(ROI)
    sensor_xyz = frame.to_xyz(llh)
    sensor_range = 1000  # metres
    sensor_scale = 1e2  # sensor parameter. Unknown?
    sensor_tol = 1e-12  # Sensor noise level pre-scaling
    noise_var = sensor_scale**2 * 1e-9  # known or learn?

    res = 20  # Output Grid resolution
    ground = make_ground(sensor_xyz, pad=2*sensor_range, res=res, frame=frame)
    n_grid = np.prod(ground.shape[:2])
    n_sensor = sensor_xyz.shape[0]
    sensor_gain = sensor_scale * res ** 2
    S = sensor_gain * sensitivity_matrix(ground, sensor_xyz, sensor_range,
                                         sensor_tol)

    # Investigate the sensor itself - we assume a white noise spatial prior
    # for now...
    G = sparse.eye(n_grid)  # Ground sparsity?
    K = S.dot(G.dot(S.T)) + noise_var * sparse.eye(n_sensor)

    y = data[:, 2]  # uranium column
    mu = S.T.dot(spsolve(K, y)).reshape(ground.shape[:2])

    # Display the prediction
    pl.figure()
    # its not quite 2d but we can approximately show as a flat image
    gx = np.mean(ground[:, :, 0], axis=1)
    gy = np.mean(ground[:, :, 1], axis=0)
    pl.imshow(mu.T, interpolation='none',
              extent=(gx[0], gx[-1], gy[0], gy[-1]))

    # Display the points used in the calculation
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ground[::2, ::2, 0].ravel(), ground[::2, ::2, 1].ravel(),
            'b.', zs=ground[::2, ::2, 2].ravel())
    ax.plot(sensor_xyz[:, 0], sensor_xyz[:, 1], 'k.', zs=sensor_xyz[:, 2])
    ax.set_zlim((-200, 250))
    ax.set_xlabel('Eastings (m)')
    ax.set_ylabel('Northings (m)')
    pl.axis('equal')
    pl.show()
    exit()


def sensitivity_matrix(ground, xyz, sensor_range, sensor_tol):
    # Exploit grid structure for spatial interactions                       
    gx = np.mean(ground[:, :, 0], axis=1)
    gy = np.mean(ground[:, :, 1], axis=0)
    # build a [sparse] sensitivity matrix
    n_sensor = xyz.shape[0]
    n_grid = ground.shape[0] * ground.shape[1]
    sensor_inds = []
    grid_inds = []
    values = []
    n = np.prod(ground.shape[:2])  # nearly a million pixels??
    index = np.arange(n, dtype=int).reshape(ground.shape[:2])
    for i in tqdm(range(n_sensor), desc="Compute sparse sensor matrix"):
        fx, fy, fz = xyz[i]

        # Exploit grid structure to shortlist possible interactions
        L = bisect_left(gx, fx - sensor_range)
        R = bisect_left(gx, fx + sensor_range)
        B = bisect_left(gy, fy - sensor_range)
        T = bisect_left(gy, fy + sensor_range)
        sensor_xyz = xyz[i][np.newaxis, np.newaxis, :]
        grid_xyz = ground[L:R, B:T]

        # Invoke sensor model
        r = np.sqrt(np.sum((grid_xyz - sensor_xyz)**2, axis=2))
        sens = np.exp(-mu_air*r) / r**2
        mask = sens > sensor_tol

        # Looks reasonable - now identify which grid elements are involved
        grid_ind = index[L:R, B:T][mask]
        sensor_ind = i + np.zeros_like(grid_ind)
        sensor_inds.append(sensor_ind)
        grid_inds.append(grid_ind)
        values.append(sens[mask])

    sensor_inds = np.hstack(sensor_inds)
    grid_inds = np.hstack(grid_inds)
    values = np.hstack(values)
    S = sparse.coo_matrix((values, (sensor_inds, grid_inds)),
                          shape=(n_sensor, n_grid))
    return S


def make_ground(xyz, pad, res, frame):

    pad = 400  # metres
    grid = 50  # metres
    xmin = np.min(xyz[:, 0]) - pad
    xmax = np.max(xyz[:, 0]) + pad
    ymin = np.min(xyz[:, 1]) - pad
    ymax = np.max(xyz[:, 1]) + pad
    res_x = int((xmax - xmin) / res)
    res_y = int((ymax - ymin) / res)

    # edges
    ex = np.linspace(xmin, xmax, res_x + 1)
    ey = np.linspace(ymin, ymax, res_y + 1)

    # centres
    cx = 0.5 * (ex[1:] + ex[:-1])
    cy = 0.5 * (ey[1:] + ey[:-1])

    # Make a grid of world points (the product image)
    px, py = np.meshgrid(cx, cy)
    
    # numpy sometimes likes [y,x], but statbadger standard is [x,y]
    px = px.T
    py = py.T

    ground = np.array([px.ravel(), py.ravel(), np.zeros(res_x * res_y)]).T

    # Send grid to ground with same lat lon using the ref frame
    ground = frame.to_llh(ground)
    ground[:, 2] = 0
    ground = frame.to_xyz(ground)

    gshape = (px.shape[0], px.shape[1], 3)
    ground = np.reshape(ground, gshape)
    return ground


@cache
def get_flightlines():
    """Collect aerial survey points."""
    log.info('Extracting flight lines...')
    sf = shapefile.Reader(data_root + 'airborne_flightlines.shp')
    survey_points = np.vstack([s.points for s in sf.shapes()])
    data = np.array(sf.records(), dtype=float)
    field = {k[0]: v for v, k in enumerate(sf.fields[1:])}
    altitude = data[:, field['altitude']]  # in feet? what?
    kthu_ind = [field['potassium_'], field['thorium_pp'], field['uranium_pp']]
    elems = data[:, kthu_ind]
    survey_points = np.hstack((survey_points, altitude[:, np.newaxis]))
    return survey_points, elems


def normed(x):
    return x / np.sqrt(np.sum(x**2))



class LocalFrame:
    """Local north east up frame."""

    def __init__(self, ROI):
        bounds = [(ROI[0], ROI[2], 0),
                  (ROI[1], ROI[2], 0),
                  (ROI[0], ROI[3], 0),
                  (ROI[1], ROI[3], 0)]
        self.wgs84 = nv.FrameE(name='WGS84')
        lon, lat, hei = np.array(bounds).T
        geo_points = self.wgs84.GeoPoint(longitude=lon, latitude=lat, z=-hei,
                                         degrees=True)
        P = geo_points.to_ecef_vector().pvector.T
        dx = normed(P[1] - P[0])
        dy = P[2] - P[0]
        dy -= dx * dy.dot(dx)
        dy = normed(dy)
        dz = np.cross(dx, dy)
        self.rotation = np.array([dx, dy, dz]).T
        self.mu = np.mean(P.dot(self.rotation), axis=0)[np.newaxis, :]
        
    def to_xyz(self, points_llh):
        lon, lat, hei = points_llh.T
        geo_points = self.wgs84.GeoPoint(longitude=lon, latitude=lat, z=-hei,
                                         degrees=True)
        ecef_points = geo_points.to_ecef_vector().pvector.T
        return (ecef_points).dot(self.rotation) - self.mu

    def to_llh(self, xyz):
        """Convert lat lon height into ecef using wgs84."""
        ecef = (xyz + self.mu).dot(self.rotation.T)
        pts = self.wgs84.ECEFvector(ecef.T)
        geo = pts.to_geo_point()
        llh = np.array([geo.longitude_deg, geo.latitude_deg, -geo.z]).T
        return llh



# We could compute 'true area' per pixel...
#np.sqrt(np.sum((ground[1,0] - ground[0,0])**2) * 
#                   np.sum((ground[0,1] - ground[0,0])**2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(formatter={'float': (lambda f: '%.2f' % f)})
    main()
    cache.close()
