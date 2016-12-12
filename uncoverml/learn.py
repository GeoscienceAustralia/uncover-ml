import logging

from uncoverml import mpiops
from uncoverml.models import modelmaps, apply_multiple_masked
from uncoverml.optimise.models import transformed_modelmaps
from uncoverml.krige.krige import krig_dict


def _join_dicts(dicts):
    if dicts is None:
        return
    d = {k: v for D in dicts for k, v in D.items()}
    return d

all_modelmaps = _join_dicts([transformed_modelmaps, modelmaps, krig_dict])

log = logging.getLogger(__name__)


def local_learn_model(x_all, targets_all, config):

    model = None
    if config.multicubist or config.multirandomforest:
        y = targets_all.observations
        model = all_modelmaps[config.algorithm](**config.algorithm_args)
        apply_multiple_masked(model.fit, (x_all, y),
                              kwargs={'fields': targets_all.fields,
                                      'parallel': True,
                                      'lat_lon': targets_all.positions})
    else:
        if mpiops.chunk_index == 0:
            y = targets_all.observations
            model = all_modelmaps[config.algorithm](**config.algorithm_args)
            apply_multiple_masked(model.fit, (x_all, y),
                                  kwargs={'fields': targets_all.fields,
                                          'lat_lon': targets_all.positions})
    return model
