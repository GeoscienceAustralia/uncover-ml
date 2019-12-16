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

def covariate_shift_targets(targets):
    def _dummy_targets(targets, label, seed=1):
        rnd = np.random.RandomState(seed)
        lons = targets.positions[:,0]
        lats = targets.positions[:,1]
        shp = targets.observations.shape
        #new_lons = rnd.uniform(np.min(lons), np.max(lons), shp)
        #new_lats = rnd.uniform(np.min(lats), np.max(lats), shp)              
        def _generate_points(old_points, limit):
            new_points = []
            while new_points < limit:
                new_point = rnd.uniform(np.min(old_points), np.max(old_points))
                if new_point not in old_points:
                    new_points.append(new_point)
            return new_points
        new_lons = _generate_points(lons, shp[0] - 1)
        new_lats = _generate_points(lats, shp[0] - 1)
        lonlats = np.column_stack([new_lons, new_lats])
        labels = np.full(shp, label)
        return Targets(lonlats, labels)

    def _label_targets(targets, label):
        labels = np.full(targets.observations.shape, label)
        return Targets(targets.positions, labels, targets.fields)

    real_targets = _label_targets(targets, 'training')
    dummy_targets = _dummy_targets(targets, 'query')
    _logger.info("Generated %s dummy targets for covariate shift", len(dummy_targets.observations))
    return Targets(np.append(dummy_targets.positions, real_targets.positions, 0),
                   np.append(dummy_targets.observations, real_targets.observations, 0),
                   dummy_targets.fields.update(real_targets.fields))

def gather_targets(targets, keep, config, node=None):
    # save_dropped_targets(config, keep, targets)
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
        result = Targets(p, y, othervals=d)
    return result


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
