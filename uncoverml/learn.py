import logging
import os

import numpy as np

from uncoverml import mpiops, diagnostics, resampling
from uncoverml.targets import Targets
from uncoverml.krige import krig_dict
from uncoverml.models import modelmaps, apply_multiple_masked
from uncoverml.optimise.models import transformed_modelmaps


_logger = logging.getLogger(__name__)
all_modelmaps = {**transformed_modelmaps, **modelmaps, **krig_dict}

# Should we put this in a model class?
# We could try and create a BootstrapEnsemble container class that 
#  takes a model as an init argument.
# Similar to multirandomforest but more generic.
def bootstrap_model(x_all, targets_all, config):
    """
    Use bootstrap resampling on target data to generate an ensemble of
    models, allowing for probabilistic predictions.

    Parameters
    ----------
    x_all
        Array of covariate data, containing data at target locations.
    targets_all : Targets
        Collection of training targets.
    n : int
        Number of models to train (each model is trained on 
        separately resampled training data).

    Returns
    -------
    list of models
        A list of uncoverml `Model`s, each trained on bootstrapped
        training data.
    """
    models = []
    for i in range(config.bootstrap_models):
        target_data = resampling.resample_by_magnitude(
            targets_all, 'observations', bins=1, bootstrap=True)
        bootstrapped_targets = Targets.from_geodataframe(target_data)
        bs_inds = [np.where(targets_all.positions == p)[0][0] 
                   for p in bootstrapped_targets.positions]
        bootstrapped_x = x_all[bs_inds]
        models.append(local_learn_model(bootstrapped_x, bootstrapped_targets, config))

        _logger.info(f"Trained model {i + 1} of {config.bootstrap_models}")

    return models


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
                config.transformed_targets_file)\
            .savefig(config.plot_target_scaling)
        

    return model
