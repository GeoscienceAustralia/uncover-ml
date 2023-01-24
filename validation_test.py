import joblib
import logging

from pathlib import Path
import os

import uncoverml.config as uml_conf
import uncoverml.validate as uml_val

from uncoverml.scripts.uncoverml import _load_data

log = logging.getLogger(__name__)


def val_test(config_file, model_file, partitions=1):
    with open(model_file, 'rb') as f:
        state_dict = joblib.load(f)

    model = state_dict["model"]

    current_config = uml_conf.Config(config_file)
    # current_config.pickle_load = False
    # current_config.target_file = current_config.oos_validation_file
    # current_config.target_property = current_config.oos_validation_property

    targets_all, x_all = _load_data(current_config, partitions)
    uml_val.oos_validate(targets_all, x_all, model, current_config)
    # uml_val.plot_feature_importance(model, x_all, targets_all, current_config)


if __name__ == '__main__':
    current_config_file = 'xgboost_fine_tune.yaml'
    current_model_file = 'iron_oxide/xgboost_fine_tune.model'
    val_test(current_config_file, current_model_file)
