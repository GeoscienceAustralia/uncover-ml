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

    config = uml_conf.Config(config_file)
    config.pickle_load = False
    config.target_file = config.oos_validation_file
    config.target_property = config.oos_validation_property

    targets_all, x_all = _load_data(config, partitions)
    uml_val.oos_validate(targets_all, x_all, model, config)

    log.info("Finished OOS validation job! Total mem = {:.1f} GB".format(_total_gb()))


if __name__ == '__main__':
    current_config_file = 'xgboost_fine_tune.yaml'
    current_model_file = 'iron_oxide/xgboost_fine_tune.model'
    val_test(current_config_file, current_model_file)
