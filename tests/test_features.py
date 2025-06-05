import os
import pickle
import tempfile

import numpy as np
import pytest
from types import SimpleNamespace

from uncoverml.features import (
    extract_subchunks,
    _image_has_targets,
    _extract_from_chunk,
    extract_features,
    transform_features,
    save_intersected_features_and_targets,
    cull_all_null_rows,
    gather_features,
    remove_missing,
)
import uncoverml.features as FU


@pytest.fixture(autouse=True)
def patch_mpiops(monkeypatch):
    monkeypatch.setattr(FU.mpiops, "chunks", 1)
    monkeypatch.setattr(FU.mpiops, "chunk_index", 0)

    class DummyComm:
        def gather(self, x, root):
            return [x]

        def allgather(self, x):
            return [x]

    monkeypatch.setattr(FU.mpiops, "comm", DummyComm())


@pytest.fixture(autouse=True)
def patch_RasterioImageSource(monkeypatch):
    class DummyRasterio:
        pass

    import uncoverml.geoio as geoio_mod
    monkeypatch.setattr(geoio_mod, "RasterioImageSource", DummyRasterio)
    yield DummyRasterio


@pytest.fixture(autouse=True)
def patch_Image_and_patch(monkeypatch):
    class DummyImage:
        def __init__(self, image_source, chunk_index, total_chunks, patchsize, template_source=None):
            self._args = (image_source, chunk_index, total_chunks, patchsize, template_source)
            self.ymin = 0
            self.ymax = 0

    monkeypatch.setattr(FU, "Image", DummyImage)

    sentinel = object()
    monkeypatch.setattr(FU.patch, "all_patches", lambda img, patchsize: sentinel)

    def dummy_patches_at_target(image, patchsize, targets):
        arr = np.arange(len(targets.observations) * 2).reshape(len(targets.observations), 2)
        mask = np.zeros_like(arr, dtype=bool)
        return SimpleNamespace(data=arr, mask=mask)

    monkeypatch.setattr(FU.patch, "patches_at_target", dummy_patches_at_target)
    yield {"sentinel": sentinel}


@pytest.fixture(autouse=True)
def patch_transforms(monkeypatch):
    class DummyTransform:
        def __init__(self, image_transforms, imputer, global_transforms, is_categorical):
            self.is_categorical = is_categorical

        def __call__(self, c):
            data = np.array([[1.0, 2.0], [3.0, 4.0]])
            mask = np.array([[False, True], [True, False]])
            return np.ma.MaskedArray(data, mask=mask)

    monkeypatch.setattr(FU.transforms, "ImageTransformSet", DummyTransform)
    yield DummyTransform


class DummyImageLike:
    def __init__(self, ymin, ymax):
        self.ymin = ymin
        self.ymax = ymax


@pytest.mark.parametrize(
    "ymin, ymax, im, expected",
    [
        (2, 5, DummyImageLike(0, 10), True),
        (10, 20, DummyImageLike(12, 15), True),
    ],
)
def test__image_has_encompass_and_edge(ymin, ymax, im, expected):
    assert _image_has_targets(ymin, ymax, im) is expected


def test__image_has_no_intersect():
    im = DummyImageLike(50, 60)
    assert _image_has_targets(0, 10, im) is False


def test_cull_all_null_rows_all_good(monkeypatch):
    fs = [
        {"anykey": np.zeros((2, 2, 2, 1))},
        {"other": np.zeros((2, 3, 3, 2))},
    ]
    result = cull_all_null_rows(fs)
    np.testing.assert_array_equal(result, np.array([True, True]))


def test_cull_all_null_rows_some_all_masked(monkeypatch):
    class CustomTransform:
        def __init__(self, image_transforms, imputer, global_transforms, is_categorical):
            self.is_categorical = is_categorical

        def __call__(self, c):
            data = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
            mask = np.array([[True, True, True], [False, False, False]])
            return np.ma.MaskedArray(data, mask=mask)

    monkeypatch.setattr(FU.transforms, "ImageTransformSet", CustomTransform)

    fs = [
        {"a": np.zeros((2, 2, 2, 1))},
        {"b": np.zeros((2, 2, 2, 1))},
    ]

    result = cull_all_null_rows(fs)
    np.testing.assert_array_equal(result, np.array([False, True]))


def test_gather_features_allgather():
    x = np.ma.MaskedArray(np.array([[1, 2], [3, 4]]), mask=[[False, True], [False, False]])
    stacked = gather_features(x, node=None)
    assert isinstance(stacked, np.ma.MaskedArray)
    np.testing.assert_array_equal(stacked.data, x.data)
    np.testing.assert_array_equal(stacked.mask, x.mask)


def test_gather_features_gather_root():
    x = np.ma.MaskedArray(np.array([[5, 6]]), mask=[[False, False]])
    stacked = gather_features(x, node=0)
    assert isinstance(stacked, np.ma.MaskedArray)
    np.testing.assert_array_equal(stacked.data, x.data)
    np.testing.assert_array_equal(stacked.mask, x.mask)


def test_remove_missing_no_mask():
    x = np.ma.MaskedArray(np.array([[1.0, 2.0], [3.0, 4.0]]), mask=[[False, False], [False, False]])
    out_x, out_classes = remove_missing(x, targets=None)
    np.testing.assert_array_equal(out_x, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert out_classes is None


def test_remove_missing_with_mask_and_targets():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    mask = np.array([[False, False], [True, False], [False, True]])
    x = np.ma.MaskedArray(data, mask=mask)
    targets = SimpleNamespace(observations=np.array([10, 20, 30]))

    out_x, out_classes = remove_missing(x, targets)
    np.testing.assert_array_equal(out_x, np.array([[1, 2]]))
    np.testing.assert_array_equal(out_classes, np.array([10]))


def test_extract_subchunks_returns_patch(patch_RasterioImageSource, patch_Image_and_patch):
    DummyRasterio = patch_RasterioImageSource
    img_src = DummyRasterio()
    result = extract_subchunks(img_src, subchunk_index=0, n_subchunks=2, patchsize=5, template_source=None)
    assert result is patch_Image_and_patch["sentinel"]


def test_extract_subchunks_requires_valid_types(patch_RasterioImageSource):
    with pytest.raises(AssertionError):
        extract_subchunks(image_source=object(), subchunk_index=0, n_subchunks=1, patchsize=3)


def test__extract_from_chunk_no_targets(monkeypatch):
    monkeypatch.setattr(FU, "_image_has_targets", lambda y_min, y_max, im: False)
    img_src = object()
    targets = SimpleNamespace(positions=np.array([[1, 1], [2, 2]]), observations=None)
    result = _extract_from_chunk(img_src, targets, chunk_index=0, total_chunks=1, patchsize=4)
    assert result is None


def test__extract_from_chunk_with_targets(monkeypatch):
    monkeypatch.setattr(FU, "_image_has_targets", lambda y_min, y_max, im: True)
    img_src = object()
    targets = SimpleNamespace(positions=np.array([[1, 1], [2, 2]]), observations=np.array([0, 1]))
    result = _extract_from_chunk(img_src, targets, chunk_index=0, total_chunks=1, patchsize=4)
    assert hasattr(result, "data") and hasattr(result, "mask")
    assert result.data.shape[0] == 2


def test_extract_features_all_outside(monkeypatch):
    monkeypatch.setattr(FU, "_extract_from_chunk", lambda isrc, tgt, ci, tc, ps: None)
    img_src = object()
    targets = SimpleNamespace(positions=np.array([[0, 0]]), observations=np.array([1]))
    with pytest.raises(ValueError):
        extract_features(img_src, targets, n_subchunks=1, patchsize=4)


def test_extract_features_some_inside(monkeypatch):
    class DummyPatch:
        def __init__(self, data, mask):
            self.data = data
            self.mask = mask

    def fake_extract(isrc, tgt, ci, tc, ps):
        if ci == 0:
            data = np.array([[1.0, 2.0], [3.0, 4.0]])
            mask = np.array([[False, False], [False, False]])
            return DummyPatch(data=data, mask=mask)
        else:
            return None

    monkeypatch.setattr(FU, "_extract_from_chunk", fake_extract)
    img_src = object()
    targets = SimpleNamespace(positions=np.array([[0, 0], [1, 1]]), observations=np.array([10, 20]))
    out = extract_features(img_src, targets, n_subchunks=2, patchsize=4)
    assert isinstance(out, np.ma.MaskedArray)
    assert out.shape == (2, 2)
    assert np.all(out.mask == False)


def test_extract_features_shape_mismatch(monkeypatch):
    class DummyPatchBad:
        def __init__(self):
            self.data = np.array([[1.0, 2.0]])
            self.mask = np.zeros_like(self.data, dtype=bool)

    def bad_extract(isrc, tgt, ci, tc, ps):
        return DummyPatchBad()

    monkeypatch.setattr(FU, "_extract_from_chunk", bad_extract)
    img_src = object()
    targets = SimpleNamespace(positions=np.array([[0, 0], [1, 1]]), observations=np.array([10, 20]))
    with pytest.raises(AssertionError):
        extract_features(img_src, targets, n_subchunks=1, patchsize=3)


def test_transform_features_standard():
    fs1 = {"f1": np.zeros((2, 2, 2, 1))}
    fs2 = {"f2": np.zeros((2, 3, 3, 2))}
    feature_sets = [fs1, fs2]

    class T0:
        def __init__(self):
            self.is_categorical = False

        def __call__(self, c):
            data = np.array([[1.0], [2.0]])
            mask = np.zeros_like(data, dtype=bool)
            return np.ma.MaskedArray(data, mask=mask)

    class T1:
        def __init__(self):
            self.is_categorical = True

        def __call__(self, c):
            data = np.array([[10.0], [20.0]])
            mask = np.zeros_like(data, dtype=bool)
            return np.ma.MaskedArray(data, mask=mask)

    transform_sets = [T0(), T1()]
    final_transform = lambda arr: arr * 2.0
    config = SimpleNamespace(
        cubist=False,
        multicubist=False,
        krige=False,
        algorithm="algoX",
        algorithm_args={},
        pickle=False,
        featurevec="does_not_matter.pkl",
    )

    x_out, good_rows = transform_features(feature_sets, transform_sets, final_transform, config)
    assert isinstance(x_out, np.ma.MaskedArray)
    np.testing.assert_array_equal(x_out.data, np.array([[2.0, 20.0], [4.0, 40.0]]))
    assert good_rows.shape == (2,)


def test_transform_features_cubist_path(tmp_path):
    fs1 = {"f1": np.zeros((2, 1, 1, 1))}
    transform_sets = []

    class DummyT:
        def __init__(self):
            self.is_categorical = False

        def __call__(self, c):
            data = np.array([[5.0], [6.0]])
            mask = np.zeros_like(data, dtype=bool)
            return np.ma.MaskedArray(data, mask=mask)

    transform_sets.append(DummyT())
    final_transform = None
    featurevec_path = tmp_path / "fv.pkl"
    config = SimpleNamespace(
        cubist=True,
        multicubist=False,
        krige=False,
        algorithm="cubie",
        algorithm_args={},
        pickle=True,
        featurevec=str(featurevec_path),
    )

    x_out, good_rows = transform_features([fs1], transform_sets, final_transform, config)
    assert isinstance(x_out, np.ma.MaskedArray)
    np.testing.assert_array_equal(x_out.data, np.array([[5.0], [6.0]]))
    assert featurevec_path.exists()
    with open(featurevec_path, "rb") as f:
        fv = pickle.load(f)
    assert isinstance(fv, dict)
    assert len(fv) == 1
    k0 = next(iter(fv.keys()))
    assert "f1" in k0


def test_transform_features_krige_ignores_final_transform():
    fs = [{"f": np.zeros((1, 1, 1, 1))}]

    class DummyT:
        def __init__(self):
            self.is_categorical = False

        def __call__(self, c):
            data = np.array([[2.0]])
            mask = np.array([[False]])
            return np.ma.MaskedArray(data, mask=mask)

    transform_sets = [DummyT()]

    def bad_final(x):
        raise RuntimeError("Should not be called")

    config = SimpleNamespace(
        cubist=False,
        multicubist=False,
        krige=True,
        algorithm="algoY",
        algorithm_args={},
        pickle=False,
        featurevec="fv.pkl",
    )

    x_out, good_rows = transform_features(fs, transform_sets, bad_final, config)
    assert isinstance(x_out, np.ma.MaskedArray)
    np.testing.assert_array_equal(x_out.data, np.array([[2.0]]))


def test_transform_and_remove_missing_together(monkeypatch):
    class CustomTransform2:
        def __init__(self, image_transforms, imputer, global_transforms, is_categorical):
            self.is_categorical = is_categorical

        def __call__(self, c):
            data = np.array([[10.0, 20.0], [30.0, 40.0]])
            mask = np.array([[True, True], [False, False]])
            return np.ma.MaskedArray(data, mask=mask)

    monkeypatch.setattr(FU.transforms, "ImageTransformSet", CustomTransform2)

    fs1 = {"f1": np.zeros((2, 1, 1, 1))}
    fs2 = {"f2": np.zeros((2, 1, 1, 1))}
    feature_sets = [fs1, fs2]
    transform_sets = [CustomTransform2(None, None, None, False), CustomTransform2(None, None, None, True)]
    final_transform = None
    config = SimpleNamespace(cubist=False, multicubist=False, krige=False, algorithm_args={}, pickle=False, featurevec="no.pkl")

    x_out, good_rows = transform_features(feature_sets, transform_sets, final_transform, config)
    np.testing.assert_array_equal(good_rows, np.array([False, True]))

    targets = SimpleNamespace(observations=np.array([1, 2]))
    x_final, classes_final = remove_missing(x_out, targets)
    np.testing.assert_array_equal(x_final, x_out.data[1].reshape(1, -1))
    np.testing.assert_array_equal(classes_final, np.array([2]))


def test_save_intersected_features_and_targets_file_writing(tmp_path, monkeypatch):
    arr = np.zeros((2, 1, 1, 1))
    feature_sets = [{"feat": arr}]

    class DummyT:
        def __init__(self):
            self.is_categorical = False

        def __call__(self, c):
            pass

    transform_sets = [DummyT()]
    targets = SimpleNamespace(positions=np.array([[0, 0], [1, 1]]), observations=np.array([100, 200]))

    rawcov = tmp_path / "rawcov.csv"
    rawcov_mask = tmp_path / "rawcov_mask.csv"
    config = SimpleNamespace(
        rawcovariates=str(rawcov),
        rawcovariates_mask=str(rawcov_mask),
        plot_covariates=False,
        target_property="dummy",
    )

    def fake_gather_features(x, node):
        return x

    monkeypatch.setattr(FU, "gather_features", fake_gather_features)

    def fake_gather(x, root):
        return [x]

    monkeypatch.setattr(FU.mpiops.comm, "gather", fake_gather)

    FU.mpiops.chunk_index = 0

    save_intersected_features_and_targets(feature_sets, transform_sets, targets, config)

    assert rawcov.exists()
    assert rawcov_mask.exists()

    data = np.loadtxt(str(rawcov), delimiter=",", skiprows=1)
    mask = np.loadtxt(str(rawcov_mask), delimiter=",", skiprows=1)
    assert data.shape == (2, 5)
    assert mask.shape == (2, 5)


def test_save_intersected_features_and_targets_no_write_when_not_root(tmp_path):
    FU.mpiops.chunk_index = 1
    fs = [{"f": np.zeros((2, 1, 1, 1))}]
    ts = [SimpleNamespace(is_categorical=False)]
    tgt = SimpleNamespace(positions=np.array([[0, 0]]), observations=np.array([7]))
    rawcov = tmp_path / "rc.csv"
    rawcov_mask = tmp_path / "rcm.csv"
    config = SimpleNamespace(rawcovariates=str(rawcov), rawcovariates_mask=str(rawcov_mask), plot_covariates=False, target_property="x")

    save_intersected_features_and_targets(fs, ts, tgt, config)
    assert not rawcov.exists()
    assert not rawcov_mask.exists()


def test_extract_subchunks_with_template_source(patch_RasterioImageSource, patch_Image_and_patch):
    DummyRasterio = patch_RasterioImageSource
    img_src = DummyRasterio()
    tpl_src = DummyRasterio()
    result = extract_subchunks(img_src, subchunk_index=0, n_subchunks=1, patchsize=3, template_source=tpl_src)
    assert result is patch_Image_and_patch["sentinel"]


def test_remove_missing_no_mask_but_with_targets():
    data = np.array([[7, 8], [9, 10]])
    mask = np.zeros_like(data, dtype=bool)
    x = np.ma.MaskedArray(data, mask=mask)
    targets = SimpleNamespace(observations=np.array([1, 2]))
    out_x, out_cls = remove_missing(x, targets)
    np.testing.assert_array_equal(out_x, data)
    np.testing.assert_array_equal(out_cls, np.array([1, 2]))
