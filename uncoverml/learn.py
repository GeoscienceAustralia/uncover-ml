import logging
import os

import numpy as np
from mpi4py import MPI

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

    item_size = MPI.DOUBLE.Get_size()

    if mpiops.chunk_index == 0:
        obs = targets_all.observations
    else:
        obs = None

    shared_x, x_win = mpiops.create_shared_array(x_all, item_size)
    shared_t, t_win = mpiops.create_shared_array(obs, item_size)

    # Resample the required amount of times and create a collection of
    # indicies for accesing shared data. We do this so we don't 
    # duplicate the data in the process of resampling on multiple 
    # nodes. Resampling is quick so should be okay.
    if mpiops.chunk_index == 0:
        inds = []
        _logger.info("Bootstrapping data %s times", config.bootstrap_models)
        for i in range(config.bootstrap_models):
            target_data = resampling.resample_by_magnitude(
                targets_all, 'observations', bins=1, bootstrap=True)
            bootstrapped_targets = Targets.from_geodataframe(target_data)
            inds.append([np.where(targets_all.positions == p)[0][0] 
                           for p in bootstrapped_targets.positions])
            _logger.info(f"Bootstrapped {i + 1} of {config.bootstrap_models}")
    else:
        inds = None
    
    _logger.info("Bootstrapping complete, training models...")

    inds = mpiops.comm.bcast(inds, root=0)
    inds = np.array_split(inds, mpiops.chunks)[mpiops.chunk_index]

    for i, ind in enumerate(inds):
        bootstrapped_x = shared_x[ind]
        bootstrapped_t = shared_t[ind]
        model = all_modelmaps[config.algorithm](**config.algorithm_args)
        apply_multiple_masked(model.fit, (bootstrapped_x, bootstrapped_t))
        models.append(model)
        print(f"Processor {mpiops.chunk_index}: trained model {i + 1} of {len(inds)}")

    shared_x = None
    x_win.Free()
    shared_t = None
    t_win.Free()

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
