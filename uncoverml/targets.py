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


def generate_dummy_targets(bounds, label, n_points, field_keys=[], seed=1):
    """
    Generate dummy points with randomly generated positions.

    Args:
        bounds (tuple of float, float, float, float): Bounding box
          to generate targets within of format 
          (xmin, ymin, xmax, ymax)
        label (str): Label to assign targets.
        n_points (int): Nuber of points to generate.
        field_keys (list of str, optional): List of keys to add to 
            `fields` property.
        seed (int, optional): Random number generator seed.

    Returns:
        Targets: A collection of randomly generated targets.
    """
    rnd = np.random.RandomState(seed)
    def _generate_points(lower, upper, limit):
        new_points = []
        while len(new_points) < limit:
            new_point = rnd.uniform(lower, upper)
            new_points.append(new_point)
        return new_points
    new_lons = _generate_points(bounds[0], bounds[2], n_points)
    new_lats = _generate_points(bounds[1], bounds[3], n_points)
    lonlats = np.column_stack([new_lons, new_lats])
    labels = np.full(lonlats.shape[0], label)
    if field_keys:
        fields = {k: np.zeros(n_points) for k in field_keys}
    else:
        fields = {}
    return Targets(lonlats, labels, fields)


def generate_covariate_shift_targets(targets, bounds):
    real_targets = label_targets(targets, 'training')
    dummy_targets = generate_dummy_targets(bounds, 'query', targets.observations.shp[0])
    _logger.info("Generated %s dummy targets for covariate shift", len(dummy_targets.observations))
    return concatenate_targets(real_targets, dummy_targets)
    
def concatenate_targets(a, b):
    """
    Concatenates two `Targets` objects.
    
    Args:
        a, b (Target): The Targets to merge.

    Returns:
        Targets: A single merged collection of targets.
    """
    new_fields = {}
    for k, v in a.fields.items():
        ar = b.fields.get(k)
        if ar is not None:
            new = np.concatenate((v, ar))
        else:
            new = v
        new_fields[k] = new

    return Targets(np.append(a.positions, b.positions, 0),
                   np.append(a.observations, b.observations, 0),
                   new_fields)

def label_targets(targets, label, backup_field=None):
    """
    Replaces target observations (the target property being trained on)
    with the given label.

    Args:
        targets (Targets): A collection of targets to label.
        label (str): The label to apply.
        backup_field (str): If present, copies the original observation
            data to the `fields` property with the provided string
            as the key.

    Return:
        Targets: The labelled targets.
    """
    labels = np.full(targets.observations.shape, label)
    if backup_field:
        targets.fields[backup_field] = targets.observations
    return Targets(targets.positions, labels, targets.fields)

def gather_targets(targets, keep, node=None):
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


def save_targets(targets, path, obs_filter=None):
    """
    Saves target positions and observation data to a CSV file.

    Args:
        targets (Targets): The targets to save.
        path (str): Path to file to save as.
        obs_filter (any, optional): If provided, will only save points
            that have this observation data.
    """
    if obs_filter:
        inds = targets.observations == obs_filter
        lons = targets.positions.T[0][inds]
        lats = targets.positions.T[1][inds]
        obs = targets.observations[inds]
    else:
        lons = targets.positions.T[0]
        lats = targets.positions.T[1]
        obs = targets.observations
    ar = np.rec.fromarrays((lons, lats, obs), names='lon,lat,obs')
    np.savetxt(path, ar, fmt='%.8f,%.8f,%s', delimiter=',', header='lon,lat,obs')


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
