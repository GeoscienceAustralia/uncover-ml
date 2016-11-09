import numpy as np
from os.path import join

from uncoverml import mpiops


class Targets:

    def __init__(self, lonlat, vals, othervals=None):
        self.fields = {}
        self.observations = vals
        self.positions = lonlat
        if othervals is not None:
            self.fields = othervals


def gather_targets(targets, keep, config, node=None):
    observations = targets.observations[keep]
    positions = targets.positions[keep]

    dropped_postions = targets.positions[~keep, :]
    dropped_observations = targets.observations[~keep].reshape(
        len(dropped_postions), -1)

    if not np.all(keep):
        np.savetxt(join(config.output_dir, 'dropped_targets.txt'),
                   np.concatenate(
                       [dropped_postions,
                        dropped_observations
                        ], axis=1),
                   delimiter=',')

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
