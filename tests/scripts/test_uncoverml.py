import os
import json
import joblib
import pickle
import tempfile
import pytest
from pathlib import Path
from click.testing import CliRunner

import uncoverml.scripts.uncoverml as cli_module
from uncoverml.config import Config


class DummyConfig:
    def __init__(self, path):
        self.algorithm = "dummy"
        self.algorithm_args = {}
        self.output_dir = tempfile.mkdtemp()
        self.cross_validate = False
        self.permutation_importance = False
        self.pickle_load = False
        self.pickled_covariates = ""
        self.pickled_targets = ""
        self.cubist = False
        self.multicubist = False
        self.feature_sets = []
        self.final_transform = type("T", (), {"global_transforms": []})
        self.mask = ""
        self.prediction_template = ""
        self.cluster = False
        self.cluster_analysis = False
        self.n_classes = 0
        self.train_data_pk = ""
        self.rawcovariates = False
        self.pca = False
        self.geotif_options = {}
        self.thumbnails = False


class T:
    def __init__(self, **kwargs):
        pass
    def get_predict_tags(self):
        return ["A"]


@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    monkeypatch.setattr(cli_module.ls.config, "Config", DummyConfig)
    monkeypatch.setattr(cli_module.ls.learn, "local_learn_model", lambda x,y,z: "model")
    monkeypatch.setattr(cli_module.ls.geoio, "export_model", lambda m,c: None)
    monkeypatch.setattr(cli_module.ls.validate, "permutation_importance", lambda m,x,y,z: None)
    monkeypatch.setattr(cli_module.ls.validate, "local_crossval", lambda x,y,z: {})
    monkeypatch.setattr(cli_module.ls.geoio, "export_crossval", lambda r,c: None)
    monkeypatch.setattr(cli_module.ls.features, "transform_features", lambda a,b,c,d: (["features"], [True]))
    monkeypatch.setattr(cli_module.ls.features, "gather_features", lambda f,node: [[1,2,3]])
    monkeypatch.setattr(cli_module.ls.targets, "gather_targets", lambda t,k,c,node: ["t1","t2"])
    monkeypatch.setattr(cli_module.ls.geoio, "load_targets", lambda shapefile,targetfield,conf: "targets")
    monkeypatch.setattr(cli_module.ls.geoio, "image_feature_sets", lambda t,c: [])
    monkeypatch.setattr(cli_module.ls.geoio, "semisupervised_feature_sets", lambda t,c: [])
    monkeypatch.setattr(cli_module.ls.geoio, "unsupervised_feature_sets", lambda c: [])
    monkeypatch.setattr(cli_module.ls.features, "remove_missing", lambda f,t=None: (f, []))
    monkeypatch.setattr(cli_module.ls.cluster, "compute_n_classes", lambda c,f: 2)
    monkeypatch.setattr(cli_module.ls.cluster, "KMeans", lambda n,f: type("K", (), {"learn": lambda self,*a,**k: None})())
    monkeypatch.setattr(cli_module.ls.geoio, "export_cluster_model", lambda m,c: None)
    monkeypatch.setattr(cli_module.ls.predict, "render_partition", lambda m,i,o,c: None)
    monkeypatch.setattr(cli_module.ls.predict, "export_pca", lambda i,o,c: None)
    monkeypatch.setattr(cli_module.ls.predict, "export_pca_fractions", lambda c: None)
    monkeypatch.setattr(cli_module.uncoverml.interface_utils, "rename_files_before_upload", lambda c,j: None)
    monkeypatch.setattr(cli_module.uncoverml.interface_utils, "create_thumbnail", lambda c,t: None)
    monkeypatch.setattr(cli_module.uncoverml.interface_utils, "calc_uncert", lambda c: None)
    monkeypatch.setattr(cli_module.uncoverml.interface_utils, "create_results_zip", lambda c: None)


def test_cli_no_args_shows_help():
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, [])
    assert "Usage" in result.output


def test_learn_requires_pipeline_file():
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["learn"])
    assert "Error" in result.output


def test_learn_with_minimal_yaml(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    with open(yaml_path, "w") as f:
        json.dump({"learning": {"algorithm": "dummy", "arguments": {}}, "output": {"directory": str(tmp_path)}}, f)
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["learn", str(yaml_path)])
    assert result.exit_code != 0


def test_superlearn_requires_args():
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["superlearn"])
    assert "Error" in result.output


def test_superlearn_minimal(tmp_path):
    yaml_path = tmp_path / "cfg2.yaml"
    with open(yaml_path, "w") as f:
        json.dump({"learning": {"algorithm": "dummy", "arguments": {}}, "output": {"directory": str(tmp_path)}}, f)
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["superlearn", str(yaml_path)])
    assert result.exit_code != 0


def test_optimise_requires_args():
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["optimise"])
    assert "Error" in result.output


def test_optimise_minimal(tmp_path):
    yaml_path = tmp_path / "cfg3.yaml"
    with open(yaml_path, "w") as f:
        json.dump({"learning": {"algorithm": "dummy", "arguments": {}}, "output": {"directory": str(tmp_path)}}, f)
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["optimise", str(yaml_path)])
    assert result.exit_code != 0


def test_cluster_requires_args():
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["cluster"])
    assert "Error" in result.output


def test_cluster_minimal(tmp_path):
    yaml_path = tmp_path / "cfg4.yaml"
    cfg = {
        "learning": {"algorithm": "dummy", "arguments": {}},
        "output": {"directory": str(tmp_path)},
        "features": []
    }
    with open(yaml_path, "w") as f:
        json.dump(cfg, f)
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["cluster", str(yaml_path)])
    assert result.exit_code != 0


def test_validate_requires_args():
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["validate"])
    assert "Error" in result.output


def test_validate_minimal(tmp_path):
    yaml_path = tmp_path / "cfg5.yaml"
    with open(yaml_path, "w") as f:
        json.dump({"learning": {"algorithm": "dummy", "arguments": {}}, "output": {"directory": str(tmp_path)}}, f)
    model_file = tmp_path / "mout.model"
    joblib.dump({"model": "m", "config": str(yaml_path)}, str(model_file))
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["validate", str(yaml_path), str(model_file), "test", "1"])
    assert result.exit_code != 0


def test_predict_minimal(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["predict", "nonexistent.model"])
    assert result.exit_code != 0


def test_pca_minimal(tmp_path):
    yaml_path = tmp_path / "cfg7.yaml"
    with open(yaml_path, "w") as f:
        json.dump({
            "learning": {"algorithm": "dummy", "arguments": {}},
            "output": {"directory": str(tmp_path)},
            "preprocessing": {"transforms": [{"whiten": {"keepdims": 1}}], "imputation": None},
            "features": []
        }, f)
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["pca", str(yaml_path)])
    assert result.exit_code != 0


def test_upload_minimal(tmp_path):
    yaml_path = tmp_path / "cfg8.yaml"
    with open(yaml_path, "w") as f:
        json.dump({"output": {"directory": str(tmp_path)}}, f)
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["upload", str(yaml_path), "job"])
    assert result.exit_code != 0


def test_clip_minimal(tmp_path, monkeypatch):
    yaml_path = tmp_path / "cfg9.yaml"
    shp = tmp_path / "t.shp"
    open(shp, "w").close()
    cfg = {
        "learning": {"algorithm": "dummy", "arguments": {}},
        "output": {"directory": str(tmp_path)},
        "target_file": str(shp),
        "feature_sets": []
    }

    with open(yaml_path, "w") as f:
        json.dump(cfg, f)

    class DummyDS:
        def __init__(self, path): pass

    monkeypatch.setattr(cli_module.gdal, "Open", lambda x: DummyDS(x))
    monkeypatch.setattr(cli_module.gdal, "Translate", lambda out, ds, projWin: ds)
    monkeypatch.setattr(cli_module.fiona, "open", lambda x: type("D", (), {"bounds": (0,0,1,1)}))
    list_file = tmp_path / "covariate_list.pkl"
    pickle.dump([{"nci_path": str(tmp_path / "a.tif")}], open(list_file, "wb"))
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["clip", str(yaml_path), "job"])
    assert result.exit_code != 0
