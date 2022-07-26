import joblib

from uncoverml import config


def EditPickle(pickle_file, new_config):
    with open(pickle_file, 'rb') as f:
        state_dict = joblib.load(f)

    model_to_use = state_dict["model"]
    config_to_use = config.Config(new_config)

    save_dict = {"model": model_to_use, "config": config_to_use}
    model_file = 'gbquantiles_new.model'
    with open(model_file, 'wb') as f:
        joblib.dump(save_dict, f)
        log.info(f"Wrote model on disc {model_file}")


if __name__ == '__main__':
    picklefile = 'gbquantile/gbquantiles.model'
    newconfig = 'gbquantiles.yaml'
    EditPickle(picklefile, newconfig)
