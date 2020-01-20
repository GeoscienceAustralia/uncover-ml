import logging
from os.path import join

import numpy as np
from uncoverml import mpiops

_logger = logging.getLogger(__name__)


class Targets:

    def __init__(self, lonlat, vals, othervals=None):
        self.fields = {}
        self.observations = vals
        self.positions = lonlat
        if othervals is not None:
            self.fields = othervals

def generate_covariate_shift_targets(targets, bounds):
    def _dummy_targets(targets, label, seed=1):
        rnd = np.random.RandomState(seed)
        lons = targets.positions[:,0]
        lats = targets.positions[:,1]
        shp = targets.observations.shape
        #new_lons = rnd.uniform(np.min(lons), np.max(lons), shp)
        #new_lats = rnd.uniform(np.min(lats), np.max(lats), shp)              
        def _generate_points(lower, upper, limit):
            new_points = []
            while len(new_points) < limit:
                #new_point = rnd.uniform(np.min(old_points), np.max(old_points))
                new_point = rnd.uniform(lower, upper)
                new_points.append(new_point)
            return new_points
        new_lons = _generate_points(bounds[0][0], bounds[0][1], shp[0])
        new_lats = _generate_points(bounds[1][0], bounds[1][1], shp[0])
        lonlats = np.column_stack([new_lons, new_lats])
        labels = np.full(shp, label)
        return Targets(lonlats, labels)

    real_targets = label_targets(targets, 'training')
    dummy_targets = _dummy_targets(targets, 'query')
    _logger.info("Generated %s dummy targets for covariate shift", len(dummy_targets.observations))
    return merge_targets(real_targets, dummy_targets)
    
def merge_targets(a, b):
    return Targets(np.append(a.positions, b.positions, 0),
                   np.append(a.observations, b.observations, 0),
                   a.fields.update(b.fields))

def label_targets(targets, label):
        labels = np.full(targets.observations.shape, label)
        return Targets(targets.positions, labels, targets.fields)

def gather_targets(targets, keep, config, node=None):
    return gather_targets_main(targets, keep, node)

def gather_targets_main(targets, keep, node):
    observations = targets.observations[keep]
    positions = targets.positions[keep]
    if node:
        y = np.ma.concatenate(mpiops.comm.gather(observations,
                                                 root=node), axis=0)
        p = np.ma.concatenate(mpiops.comm.gather(positions,
                                                 root=node),
                              axis=0)
        d = {}
        keys = sorted(list(targets.fields.keys()))
        for k in keys:
            d[k] = np.ma.concatenate(
                mpiops.comm.gather(targets.fields[k][keep], root=node), axis=0)
        result = Targets(p, y, othervals=d)
    else:
        y = np.ma.concatenate(mpiops.comm.allgather(
            observations), axis=0)
        p = np.ma.concatenate(mpiops.comm.allgather(positions),
                              axis=0)
        d = {}
        keys = sorted(list(targets.fields.keys()))
        for k in keys:
            d[k] = np.ma.concatenate(
                mpiops.comm.allgather(targets.fields[k][keep]), axis=0)
        return Targets(p, y, othervals=d)


def save_dummy_targets(targets, config):
    dummies = []
    for obs, pos in zip(targets.observations, targets.positions):
        if obs == 'query':
            dummies.append(pos)
    np.savetxt(config.shiftmap_points, np.array(dummies), fmt='%.8f', delimiter=',')


def save_dropped_targets(config, keep, targets):
    if not np.all(keep):
        dropped_postions = targets.positions[~keep, :]
        dropped_observations = targets.observations[~keep].reshape(
            len(dropped_postions), -1)
        np.savetxt(config.dropped_targets_file,
                   np.concatenate(
                       [dropped_postions,
                        dropped_observations
                        ], axis=1),
                   delimiter=',')
