import logging
import os

import numpy as np

from uncoverml import mpiops, diagnostics
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
        if config.multirandomforest:
            rf_dicts = model._randomforests
            rf_dicts = mpiops.comm.gather(rf_dicts, root=0)
            mpiops.comm.barrier()
            if mpiops.chunk_index == 0:
                for rf in rf_dicts:
                    model._randomforests.update(rf)
    else:
        if mpiops.chunk_index == 0:
            y = targets_all.observations
            model = all_modelmaps[config.algorithm](**config.algorithm_args)
            apply_multiple_masked(model.fit, (x_all, y),
                                  kwargs={'fields': targets_all.fields,
                                          'lon_lat': targets_all.positions})

    # Save transformed targets for diagnostics
    if mpiops.chunk_index == 0 and hasattr(model, 'target_transform'):
        hdr = 'nontransformed,transformed'
        y = targets_all.observations
        y_t = model.target_transform.transform(y)
        np.savetxt(config.transformed_targets_file, X=np.column_stack((y, y_t)),
                   delimiter=',', header=hdr, fmt='%.4e')

        if config.plot_target_scaling:
            diagnostics.plot_target_scaling(
                config.transformed_targets_file).savefig(config.plot_target_scaling)
        

    return model
