import tempfile

import numpy as np
from os.path import join
import logging

from uncoverml import mpiops, resampling
log = logging.getLogger(__name__)

class Targets:

    def __init__(self, lonlat, vals, othervals=None):
        self.fields = {}
        self.observations = vals
        self.positions = lonlat
        if othervals is not None:
            self.fields = othervals


def gather_targets(targets, keep, config, node=None):
    save_dropped_targets(config, keep, targets)
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
        if not np.all(keep):
            np.savetxt(join(config.output_dir, 'dropped_targets.txt'),
                       np.concatenate(
                           [dropped_postions,
                            dropped_observations
                            ], axis=1),
                       delimiter=',')


def resample_shapefile(config):
    shapefile = config.target_file

    if not config.resample:
        return shapefile
    else:  # sample shapefile
        log.info('Stripping shapefile of unnecessary attributes')
        temp_shapefile = tempfile.mktemp(suffix='.shp', dir=config.output_dir)

        if config.resample == 'value':
            log.info("resampling shape file "
                     "based on '{}' values".format(config.target_property))
            resampling.resample_shapefile(shapefile,
                                          temp_shapefile,
                                          target_field=config.target_property,
                                          **config.resample_args
                                          )
        else:
            assert config.resample == 'spatial', \
                "resample must be 'value' or 'spatial'"
            log.info("resampling shape file spatially")

            resampling.resample_shapefile_spatially(
                shapefile, temp_shapefile,
                target_field=config.target_property,
                **config.resample_args)

        return temp_shapefile