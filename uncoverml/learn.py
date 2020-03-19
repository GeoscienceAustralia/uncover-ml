import logging
import os
import pickle
import itertools

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
        pos = targets_all.positions
    else:
        obs = None
        pos = None

    shared_x, x_win = mpiops.create_shared_array(x_all, item_size)
    shared_t, t_win = mpiops.create_shared_array(obs, item_size)
    shared_p, p_win = mpiops.create_shared_array(pos, item_size)

    if config.bootstrap_pickle is not None \
            and os.path.exists(os.path.abspath(config.bootstrap_pickle)):
        if mpiops.chunk_index == 0:
            _logger.info("Loading bootstrapped data views from file...")
            with open(os.path.abspath(config.bootstrap_pickle), 'rb') as f:
                inds = pickle.load(f)
        else:
            inds = None
        inds = mpiops.comm.bcast(inds, root=0)
        _logger.info(f"Loaded {len(inds)} data views")
        inds = np.array_split(inds, mpiops.chunks)[mpiops.chunk_index]
    else:
       _logger.info("Bootstrapping data %s times", config.bootstrap_models)
       iterations = len(np.array_split(range(config.bootstrap_models), mpiops.chunks)[mpiops.chunk_index])
       inds = []
       for i in range(iterations):
           inds.append(resampling.bootstrap_data_indicies(shared_p, max(shared_p.shape)))
           print(f"Processor {mpiops.chunk_index}: bootstrapped {i + 1} of {iterations}")

       if config.bootstrap_pickle:
           all_inds = mpiops.comm.gather(inds, root=0)
           if mpiops.chunk_index == 0:
               all_inds = list(itertools.chain.from_iterable(all_inds))
               with open(os.path.abspath(config.bootstrap_pickle), 'wb') as f:
                   pickle.dump(all_inds, f)
               _logger.info("Pickled bootstrapped data views...")
    
       _logger.info("Bootstrapping complete, training models...")

    mpiops.comm.barrier()

    for i, ind in enumerate(inds):
        bootstrapped_x = shared_x[ind]
        bootstrapped_t = shared_t[ind]
        model = all_modelmaps[config.algorithm](**config.algorithm_args)
        apply_multiple_masked(model.fit, (bootstrapped_x, bootstrapped_t))
        models.append(model)
        print(f"Processor {mpiops.chunk_index}: trained model {i + 1} of {len(inds)}")

    mpiops.comm.barrier()
    _logger.info(f"Complete! {config.bootstrap_models} trained")
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
