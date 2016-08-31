import logging

from uncoverml import mpiops
from uncoverml.models import modelmaps, apply_multiple_masked

log = logging.getLogger(__name__)


def local_learn_model(x_all, targets_all, config):
    model = modelmaps[config.algorithm](**config.algorithm_args)
    y = targets_all.observations

    if mpiops.chunk_index == 0:
        apply_multiple_masked(model.fit, (x_all, y),
                              kwargs={'fields': targets_all.fields})
    model = mpiops.comm.bcast(model, root=0)
    return model
