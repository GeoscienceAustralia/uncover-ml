import logging

from uncoverml import mpiops
from uncoverml.krige import krig_dict
from uncoverml.models import modelmaps, apply_multiple_masked
from uncoverml.targets import Targets
from uncoverml.optimise.models import transformed_modelmaps


log = logging.getLogger(__name__)
all_modelmaps = {**transformed_modelmaps, **modelmaps, **krig_dict}


def local_learn_model(x_all, targets_all: Targets, config):

    model = None
    if config.multicubist or config.multirandomforest:
        y = targets_all.observations
        weights = targets_all.weights
        model = all_modelmaps[config.algorithm](**config.algorithm_args)
        apply_multiple_masked(model.fit, (x_all, y),
                              ** {'fields': targets_all.fields,
                                  'parallel': True,
                                  'sample_weight': weights,
                                  'lon_lat': targets_all.positions})
    else:
        if mpiops.chunk_index == 0:
            y = targets_all.observations
            weights = targets_all.weights
            model = all_modelmaps[config.algorithm](**config.algorithm_args)
            apply_multiple_masked(model.fit, (x_all, y),
                                  ** {'fields': targets_all.fields,
                                      'sample_weight': weights,
                                      'lon_lat': targets_all.positions})
    return model
