import logging
import os

import numpy as np

from uncoverml import mpiops
from uncoverml.krige import krig_dict
from uncoverml.models import modelmaps, apply_multiple_masked
from uncoverml.optimise.models import transformed_modelmaps


log = logging.getLogger(__name__)
all_modelmaps = {**transformed_modelmaps, **modelmaps, **krig_dict}


def local_learn_model(x_all, targets_all, config):

    model = None
    if config.multicubist or config.multirandomforest:
        y = targets_all.observations
        model = all_modelmaps[config.algorithm](**config.algorithm_args)
        apply_multiple_masked(model.fit, (x_all, y),
                              kwargs={'fields': targets_all.fields,
                                      'parallel': True,
                                      'lon_lat': targets_all.positions})
    else:
        if mpiops.chunk_index == 0:
            y = targets_all.observations
            model = all_modelmaps[config.algorithm](**config.algorithm_args)
            apply_multiple_masked(model.fit, (x_all, y),
                                  kwargs={'fields': targets_all.fields,
                                          'lon_lat': targets_all.positions})

    # Save transformed targets for diagnostics
    if hasattr(model, 'target_transform'):
        hdr = 'nontransformed,transformed'
        y = targets_all.observations
        y_t = model.target_transform.transform(y)
        np.savetxt(config.transformed_targets_file, X=np.column_stack((y, y_t)),
                   delimiter=',', header=hdr, fmt='%.4e')

    return model
