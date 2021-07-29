import logging
from os.path import join

import numpy as np
from uncoverml import mpiops

log = logging.getLogger(__name__)


class Targets:

    def __init__(self, lonlat, vals, groups, weights, othervals=None):
        self.fields = {}
        self.observations = vals
        self.positions = lonlat
        self.groups = groups
        self.weights = weights
        if othervals is not None:
            self.fields = othervals


def gather_targets(targets, keep, config, node=None):
    # save_dropped_targets(config, keep, targets)
    return gather_targets_main(targets, keep, node)


def gather_targets_main(targets, keep, node):
    observations = targets.observations[keep]
    positions = targets.positions[keep]
    groups = targets.groups[keep]
    weights = targets.weights[keep]
    if node:
        y = np.ma.concatenate(mpiops.comm.gather(observations, root=node), axis=0)
        p = np.ma.concatenate(mpiops.comm.gather(positions, root=node), axis=0)
        g = np.ma.concatenate(mpiops.comm.gather(groups, root=node), axis=0)
        w = np.ma.concatenate(mpiops.comm.gather(weights, root=node), axis=0)
        d = {}
        keys = sorted(list(targets.fields.keys()))
        for k in keys:
            d[k] = np.ma.concatenate(mpiops.comm.gather(targets.fields[k][keep], root=node), axis=0)
        result = Targets(p, y, g, w, othervals=d)
    else:
        y = np.ma.concatenate(mpiops.comm.allgather(observations), axis=0)
        p = np.ma.concatenate(mpiops.comm.allgather(positions), axis=0)
        g = np.ma.concatenate(mpiops.comm.allgather(groups), axis=0)
        w = np.ma.concatenate(mpiops.comm.allgather(weights), axis=0)
        d = {}
        keys = sorted(list(targets.fields.keys()))
        for k in keys:
            d[k] = np.ma.concatenate(mpiops.comm.allgather(targets.fields[k][keep]), axis=0)
        result = Targets(p, y, g, w, othervals=d)
    return result


def save_dropped_targets(config, keep, targets):
    if not np.all(keep):
        dropped_postions = targets.positions[~keep, :]
        dropped_observations = targets.observations[~keep].reshape(
            len(dropped_postions), -1)
        np.savetxt(join(config.output_dir, 'dropped_targets.txt'),
                   np.concatenate(
                       [dropped_postions,
                        dropped_observations
                        ], axis=1),
                   delimiter=',')
