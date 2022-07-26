import joblib

from uncoverml import config


def EditPickle(pickle_file, delete_list):
    with open(model_or_cluster_file, 'rb') as f:
        state_dict = joblib.load(f)

    model = state_dict["model"]
    config = state_dict["config"]

    save_dict = {"model": model, "config": config}
    model_file = pickle_file
    with open(model_file, 'wb') as f:
        joblib.dump(save_dict, f)
        log.info(f"Wrote model on disc {model_file}")


if __name__ == '__main__':
    picklefile = 'gbquantiles.model'
    delete_list = ['placeholder']