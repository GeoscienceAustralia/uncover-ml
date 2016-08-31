import numpy as np

from uncoverml import mpiops


class Targets:

    def __init__(self, lonlat, vals, othervals=None):
        self.fields = {}
        self.observations = vals
        self.positions = lonlat
        if othervals is not None:
            self.fields = othervals


def gather_targets(targets):
    y = np.ma.concatenate(mpiops.comm.allgather(targets.observations), axis=0)
    p = np.ma.concatenate(mpiops.comm.allgather(targets.positions), axis=0)
    d = {}
    keys = sorted(list(targets.fields.keys()))
    for k in keys:
        d[k] = np.ma.concatenate(mpiops.comm.allgather(targets.fields[k]),
                                 axis=0)
    result = Targets(p, y, othervals=d)
    return result
