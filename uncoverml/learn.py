"""
Handles calling learning methods on models.
"""
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

def local_learn_model(x_all, targets_all, config):
    """
    Trains a model. Handles special case of parallel models.

    Parameters
    ----------
    x_all : np.ndarray
        All covariate data, shape (n_samples, n_features), sorted using
        X, Y of target positions.
    targets_all : np.ndarray
        All target data, shape (n_samples), sorted using X, Y of
        target positions.
    config : :class:`~uncoverml.config.Config`
        Config object.

    Returns
    -------
    :class:`~uncoverml.model.Model`
        A trained Model.
    """
    mpiops.comm_world.barrier()
    model = None
    if config.target_weight_property:
        weights = targets_all.fields[config.target_weight_property]
    else:
        weights = None
    # Handle models that can be trained in parallel
    if config.multicubist or config.multirandomforest or config.bootstrap:
        y = targets_all.observations
        model = all_modelmaps[config.algorithm](**config.algorithm_args)
        apply_multiple_masked(model.fit, (x_all, y), fields=targets_all.fields,
                              lon_lat=targets_all.positions,
                              sample_weight=weights)
        # Special case: for MRF we need to gather the forests from each
        # process and cache them in the model
        if config.multirandomforest:
            rf_dicts = model._randomforests
            rf_dicts = mpiops.comm_world.gather(rf_dicts, root=0)
            mpiops.comm_world.barrier()
            if mpiops.leader_world:
                for rf in rf_dicts:
                    model._randomforests.update(rf)
    # Single-threaded models
    else:
        if mpiops.leader_world:
            y = targets_all.observations
            model = all_modelmaps[config.algorithm](**config.algorithm_args)
            apply_multiple_masked(model.fit, (x_all, y), 
                                  fields=targets_all.fields, lon_lat=targets_all.positions,
                                  sample_weight=weights)

    # Save transformed targets for diagnostics
    if mpiops.leader_world and hasattr(model, 'target_transform'):
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
