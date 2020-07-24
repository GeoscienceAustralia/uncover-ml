import logging
from os.path import join

import numpy as np
import pandas as pd
import geopandas as gpd
from uncoverml import mpiops, geoio

_logger = logging.getLogger(__name__)


class Targets:
    def __init__(self, lonlat, vals, othervals=None):
        self.fields = {}
        self.observations = vals
        self.positions = lonlat
        if othervals is not None:
            self.fields = othervals

    def to_geodataframe(self):
        """
        Returns a copy of the targets as a geopandas dataframe.

        Returns
        -------
        geopandas.GeoDataFrame
        """
        df = pd.DataFrame({
            'observations': self.observations,
            'lon': self.positions[:,0],
            'lat': self.positions[:,1],
        })
        for k, v in self.fields.items():
            df[k] = v
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        return gdf

    @classmethod
    def from_geodataframe(cls, gdf, observations_field='observations'):
        """
        Returns a `Targets` object from a geopandas dataframe. 
        One column  will be taken as the main 'observations' field. All
        remaining non-geometry columns will be stored in the `fields`
        property.

        Parameters
        ----------
        observations_field : str
            Name of the column in the dataframe that is the main 
            target observation (the field to train on).

        Returns
        -------
        Targets
        """
        obs = gdf[observations_field]
        positions = np.asarray([[g.x, g.y] for g in gdf.geometry])
        fields = [c for c in gdf.columns if c != observations_field and c != 'geometry']
        return cls(positions, obs, fields)


def drop_target_values(targets, drop_values):
    """
    Drop rows from the targets for observations that have 
    particular values.

    Parameters
    ----------
    targets: `uncoverml.targets.Targets`
        An `uncoverml.targets.Targets` object that has been loaded from
        a shapefile.
    target_drop_values: list of any
        A list of values where if target observation is equal to value
        that row is dropped and also won't be intersected with the 
        covariates.
    """
    if drop_values is not None:
        for tdv in drop_values:
            if np.dtype(type(tdv)).kind != targets.observations.dtype.kind:
                raise TypeError(
                    f"Value '{tdv}' to drop from target property field has dtype "
                    f"'{np.dtype(type(tdv))} which is incompatible with target property "
                    f"dtype {targets.observations.dtype}. Change this value to a compatible dtype.")
        keep = ~np.isin(targets.observations, drop_values)
        targets.observations = targets.observations[keep]
        targets.positions = targets.positions[keep]
        for k, v in targets.fields.items():
            targets.fields[k] = v[keep]
        _logger.info(f"Dropped {np.count_nonzero(~keep)} rows from targets that contained values "
                     f"{drop_values}")
    return targets


def generate_dummy_targets(bounds, label, n_points, field_keys=[], seed=1):
    """
    Generate dummy points with randomly generated positions. Points
    are generated on node 0 and distributed to other nodes if running
    in parallel.

    Parameters
    ----------
    bounds : tuple of float
        Bounding box to generate targets within, of format
        (xmin, ymin, xmax, ymax).
    label : str 
        Label to assign generated targets.
    n_points : int
        Number of points to generate
    field_keys : list of str, optional
        List of keys to add to `fields` property.
    seed : int, optional
        Random number generator seed.

    Returns
    -------
    Targets
        A collection of randomly generated targets.
    """
    if mpiops.chunk_index == 0:
        rnd = np.random.RandomState(seed)
        def _generate_points(lower, upper, limit):
            new_points = []
            while len(new_points) < limit:
                new_point = rnd.uniform(lower, upper)
                new_points.append(new_point)
            return new_points
        new_lons = _generate_points(bounds[0], bounds[2], n_points)
        new_lats = _generate_points(bounds[1], bounds[3], n_points)
        lonlats = np.column_stack([sorted(new_lons), sorted(new_lats)])
        labels = np.full(lonlats.shape[0], label)
        if field_keys:
            fields = {k: np.zeros(n_points) for k in field_keys}
        else:
            fields = {}
        _logger.info("Generated %s dummy targets", len(lonlats))
        # Split for distribution
        lonlats = np.array_split(lonlats, mpiops.chunks)
        labels = np.array_split(labels, mpiops.chunks)
        split_fields = {k: np.array_split(v, mpiops.chunks) for k, v in fields.items()}
        fields = [{k: v[i] for k, v in split_fields.items()} for i in range(mpiops.chunks)]
    else:
        lonlats, labels, fields = None, None, None

    lonlats = mpiops.comm.scatter(lonlats, root=0)
    labels = mpiops.comm.scatter(labels, root=0)
    fields = mpiops.comm.scatter(fields, root=0)
    
    return Targets(lonlats, labels, fields)


def generate_covariate_shift_targets(targets, bounds, n_points):
    real_targets = label_targets(targets, 'training')
    dummy_targets = generate_dummy_targets(bounds, 'query', n_points)
    return merge_targets(real_targets, dummy_targets)
 

def merge_targets(a, b):
    """
    Merges two `Targets` objects. They will be sorted the canonical
    uncover-ml way: lexically by position (y, x).
    
    Args:
        a, b (Target): The Targets to merge.

    Returns:
        Targets: A single merged collection of targets.
    """
    new_fields = {}
    for k, v in a.fields.items():
        ar = b.fields.get(k, np.zeros(len(v)))
        if ar is not None:
            new = np.concatenate((v, ar))
        # Pad out with zeros so len(v(a+b)) == len(a) + len(b) as it's
        #  execpted there's the same number of values per key as there
        #  is positions/observations.
        total_len = len(a.positions) + len(b.positions)
        new = np.pad(new, (0, abs(len(new) - total_len)), constant_values=0)
        new_fields[k] = new

    pos = np.append(a.positions, b.positions, 0)
    obs = np.append(a.observations, b.observations, 0)

    ordind = np.lexsort(pos.T)
    pos = pos[ordind]
    obs = obs[ordind]
    for k, v in new_fields.items():
        new_fields[k] = v[ordind]

    return Targets(pos, obs, new_fields)


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
    if node is not None:
        y = mpiops.comm.gather(observations, root=node)
        p = mpiops.comm.gather(positions, root=node)
        d = {}
        keys = sorted(list(targets.fields.keys()))
        for k in keys:
            d[k] = mpiops.comm.gather(targets.fields[k][keep], root=node)
            
        if mpiops.chunk_index == node:
            y = np.ma.concatenate(y, axis=0)
            p = np.ma.concatenate(p, axis=0)
            for k in keys:
                d[k] = np.ma.concatenate(d[k], axis=0)
            result = Targets(p, y, othervals=d)
        else:
            result = None
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
