# tests/test_config.py
import os
import csv
import yaml
import pytest
import logging

import uncoverml.config as cm


class DummyImputer:
    def __init__(self, **kwargs):
        self.params = kwargs

class DummyImageTransform:
    def __init__(self, **kwargs):
        self.params = kwargs

class DummyGlobalTransform:
    def __init__(self, **kwargs):
        self.params = kwargs

class DummyImageTransformSet:
    def __init__(self, image_transforms, imputer, global_transforms, is_categorical):
        self.image_transforms = image_transforms
        self.imputer = imputer
        self.global_transforms = global_transforms
        self.is_categorical = is_categorical

@pytest.fixture(autouse=True)
def patch_transforms(monkeypatch):
    monkeypatch.setattr(cm.transforms, "MeanImputer", DummyImputer)
    monkeypatch.setattr(cm.transforms, "GaussImputer", DummyImputer)
    monkeypatch.setattr(cm.transforms, "NearestNeighboursImputer", DummyImputer)
    monkeypatch.setattr(cm.transforms, "OneHotTransform", DummyImageTransform)
    monkeypatch.setattr(cm.transforms, "RandomHotTransform", DummyImageTransform)
    monkeypatch.setattr(cm.transforms, "CentreTransform", DummyGlobalTransform)
    monkeypatch.setattr(cm.transforms, "StandardiseTransform", DummyGlobalTransform)
    monkeypatch.setattr(cm.transforms, "LogTransform", DummyGlobalTransform)
    monkeypatch.setattr(cm.transforms, "SqrtTransform", DummyGlobalTransform)
    monkeypatch.setattr(cm.transforms, "WhitenTransform", DummyGlobalTransform)
    monkeypatch.setattr(cm.transforms, "ImageTransformSet", DummyImageTransformSet)
    monkeypatch.setitem(cm._imputers, "mean", DummyImputer)
    monkeypatch.setitem(cm._imputers, "gaus", DummyImputer)
    monkeypatch.setitem(cm._imputers, "nn", DummyImputer)


def test_parse_transform_set_with_imputer_and_transforms():
    transform_dict = [
        {"onehot": {"param1": 10}},
        {"centre": {"param2": 5}}
    ]
    imputer_string = "mean"
    n_images = 3

    image_transforms, imputer, global_transforms = cm._parse_transform_set(transform_dict, imputer_string, n_images)

    assert isinstance(imputer, DummyImputer)
    assert imputer.params == {}
    assert len(image_transforms) == 1
    sublist = image_transforms[0]
    assert len(sublist) == 3
    for t in sublist:
        assert isinstance(t, DummyImageTransform)
        assert t.params == {"param1": 10}
    assert len(global_transforms) == 1
    assert isinstance(global_transforms[0], DummyGlobalTransform)
    assert global_transforms[0].params == {"param2": 5}


def test_parse_transform_set_with_no_imputer_and_empty_dict():
    transform_dict = None
    imputer_string = "invalid_key"
    image_transforms, imputer, global_transforms = cm._parse_transform_set(transform_dict, imputer_string, n_images=2)
    assert image_transforms == []
    assert imputer is None
    assert global_transforms == []


def test_feature_set_config_single_path(tmp_path):
    tif_path = tmp_path / "fileA.tif"
    tif_path.write_text("dummy")
    d = {
        "name": "featA",
        "type": "ordinal",
        "files": [
            {"path": str(tif_path)}
        ],
    }
    fsc = cm.FeatureSetConfig(d)
    expected = [str(tif_path.resolve())]
    assert fsc.files == expected
    ts = fsc.transform_set
    assert isinstance(ts, DummyImageTransformSet)
    assert ts.image_transforms == []
    assert ts.imputer is None
    assert ts.global_transforms == []
    assert ts.is_categorical is False


def test_feature_set_config_directory_and_list(tmp_path):
    dir_path = tmp_path / "tifdir"
    dir_path.mkdir()
    tif1 = dir_path / "b.tif"
    tif2 = dir_path / "a.tif"
    tif1.write_text("dummy1")
    tif2.write_text("dummy2")

    tif3 = tmp_path / "c.tif"
    tif3.write_text("dummy3")
    list_csv = tmp_path / "list.csv"
    with open(list_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([str(tif3)])
        writer.writerow(["#commented"])
        writer.writerow(["    "])
    d = {
        "name": "featB",
        "type": "categorical",
        "files": [
            {"directory": str(dir_path)},
            {"list": str(list_csv)}
        ],
    }
    fsc = cm.FeatureSetConfig(d)
    expected = sorted([str(tif1.resolve()), str(tif2.resolve()), str(tif3.resolve())], key=str.lower)
    assert fsc.files == expected
    ts = fsc.transform_set
    assert isinstance(ts, DummyImageTransformSet)
    assert ts.image_transforms == []
    assert ts.imputer is None
    assert ts.global_transforms == []
    assert ts.is_categorical is True


def test_feature_set_config_invalid_type_logs_warning(caplog, tmp_path):
    caplog.set_level(logging.WARNING)
    tif_path = tmp_path / "x.tif"
    tif_path.write_text("dummy")
    d = {
        "name": "featC",
        "type": "unknown_type",
        "files": [
            {"path": str(tif_path)}
        ]
    }
    fsc = cm.FeatureSetConfig(d)
    assert "Feature set type must be ordinal or categorical" in caplog.text


def test_config_minimal_yaml(tmp_path):
    outdir = tmp_path / "out"
    yaml_dict = {
        "output": {"directory": str(outdir)},
        "features": [
            {
                "name": "feat1",
                "type": "ordinal",
                "files": [{"path": str(tmp_path / "f1.tif")}]
            }
        ]
    }
    tif_file = tmp_path / "f1.tif"
    tif_file.write_text("dummy")

    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    cfg = cm.Config(str(yaml_path))
    assert cfg.name == "config"
    assert os.path.isdir(str(outdir))
    assert len(cfg.feature_sets) == 1
    fsc = cfg.feature_sets[0]
    assert fsc.name == "feat1"
    assert fsc.type == "ordinal"


def test_config_missing_features_section_raises(tmp_path):
    yaml_dict = {
        "output": {"directory": str(tmp_path / "out2")}
    }
    yaml_path = tmp_path / "badconfig.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    with pytest.raises(KeyError):
        cm.Config(str(yaml_path))


def test_config_invalid_resample_raises(tmp_path):
    outdir = tmp_path / "out3"
    yaml_dict = {
        "output": {"directory": str(outdir)},
        "targets": {
            "file": "dummy.txt",
            "resample": {}
        }
    }
    yaml_path = tmp_path / "badresample.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    with pytest.raises(KeyError):
        cm.Config(str(yaml_path))
