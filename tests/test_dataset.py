from dataclasses import replace
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from ct_pipeline.dataset import (CTSlabDataset, PreprocessConfig, make_dataloader,
                                 read_volume, restore_volume)
from ct_pipeline.cases import load_cases
from ct_pipeline.splits import create_splits


@pytest.mark.parametrize("slab", [1, 3, 5, 11])
def test_master_and_mask_alignment(write_case, slab):
    case = write_case()
    dataset = CTSlabDataset([case], PreprocessConfig(slab_size=slab, size_hw=None))
    sample = dataset[6]
    assert sample["image_slab"].shape == (1, slab, 6, 8)
    raw = nib.load(case.image).get_fdata()
    expected = np.clip((raw[:, :, 6].T + 1000) / 1400, 0, 1)
    np.testing.assert_allclose(sample["image_slab"][0, slab // 2], expected, atol=1e-6)
    np.testing.assert_array_equal(dataset[1]["master_mask"][0], nib.load(case.mask).get_fdata()[:, :, 1].T)
    np.testing.assert_allclose(sample["z_offsets_mm"], np.arange(-slab // 2 + 1, slab // 2 + 1) * 5)
    assert set(torch.unique(sample["master_mask"]).tolist()) <= {0, 1}


@pytest.mark.parametrize("master", [0, 12])
def test_boundary_padding(write_case, master):
    dataset = CTSlabDataset([write_case()], PreprocessConfig(size_hw=None))
    sample = dataset[master]
    indices = np.clip(np.arange(master - 5, master + 6), 0, 12)
    np.testing.assert_array_equal(sample["slice_indices"], indices)
    np.testing.assert_allclose(sample["z_offsets_mm"], (indices - master) * 5)
    if master == 0:
        assert torch.equal(sample["image_slab"][0, 0], sample["image_slab"][0, 5])
    else:
        assert torch.equal(sample["image_slab"][0, 5], sample["image_slab"][0, -1])


def test_short_volume_padding(write_case):
    case = write_case(shape=(8, 6, 2))
    sample = CTSlabDataset([case], PreprocessConfig(intensity_encoding="hu"))[0]
    assert sample["image_slab"].shape == (1, 11, 256, 256)
    assert sample["slice_indices"].tolist() == [0] * 6 + [1] * 5


def test_resize_nearest_mask_and_batches(write_case):
    case = write_case()
    dataset = CTSlabDataset([case])
    sample = dataset[1]
    assert sample["master_mask"].shape == (1, 256, 256)
    assert set(torch.unique(sample["master_mask"]).tolist()) == {0, 1}
    # This integer scale factor has an exact nearest-neighbor round trip.
    image, labels, metadata = read_volume(case, PreprocessConfig(size_hw=(12, 16)))
    restored = restore_volume(labels, metadata)
    np.testing.assert_array_equal(restored, nib.load(case.mask).get_fdata())
    for size in (1, 2):
        batch = next(iter(make_dataloader(dataset, batch_size=size)))
        assert batch["image_slab"].shape == (size, 1, 11, 256, 256)
        assert len(batch["metadata"]) == size


@pytest.mark.parametrize("affine", [
    np.diag([-1.0, -2.0, 5.0, 1.0]),
    np.diag([1.0, 2.0, -5.0, 1.0]),
    np.array([[0, 0, 5, 11], [-1, 0, 0, 8], [0, 2, 0, -3], [0, 0, 0, 1]], dtype=float),
])
def test_orientation_restored_to_original(write_case, affine):
    case = write_case(affine=affine)
    _, labels, metadata = read_volume(case, PreprocessConfig(size_hw=None))
    assert metadata["canonical_orientation"] == ["R", "A", "S"]
    np.testing.assert_array_equal(restore_volume(labels, metadata), nib.load(case.mask).get_fdata())
    dataset = CTSlabDataset([case], PreprocessConfig(size_hw=None))
    assert len(dataset) == labels.shape[0]


def test_oblique_positions_and_unit_conversion(write_case):
    affine = np.eye(4)
    affine[:3, :3] = [[0.001, 0, 0.003], [0, 0.002, 0], [0, 0, 0.004]]
    affine[:3, 3] = [0.1, -0.2, 0.3]
    sample = CTSlabDataset([write_case(affine=affine, unit="meter")], PreprocessConfig(size_hw=None))[6]
    np.testing.assert_allclose(sample["z_offsets_mm"], np.arange(-5, 6) * 5, atol=1e-5)
    expected = nib.affines.apply_affine(affine * np.array([[1000], [1000], [1000], [1]]), [3.5, 2.5, 6])
    np.testing.assert_allclose(sample["slice_positions_mm"][5], expected, atol=1e-4)


@pytest.mark.parametrize("kind", ["affine", "shape", "labels", "ct_nan", "mask_inf", "normalized"])
def test_bad_data_rejected(write_case, kind):
    case = write_case()
    if kind == "affine":
        mask = nib.load(case.mask)
        affine = mask.affine.copy()
        affine[0, 3] += 2
        nib.save(nib.Nifti1Image(mask.get_fdata(), affine, mask.header), case.mask)
    elif kind == "shape":
        case = write_case(mask=np.zeros((7, 6, 13), dtype=np.uint8))
    elif kind in ("labels", "mask_inf"):
        mask = np.zeros((8, 6, 13), dtype=np.float32)
        mask[0, 0, 0] = 2 if kind == "labels" else np.inf
        case = write_case(mask=mask)
    else:
        ct = np.zeros((8, 6, 13), dtype=np.float32)
        if kind == "ct_nan":
            ct[0, 0, 0] = np.nan
        case = write_case(ct=ct)
    with pytest.raises(ValueError):
        read_volume(case, PreprocessConfig())


def test_scaling_and_encoding(write_case):
    case = write_case()
    source = nib.load(case.image)
    original = source.get_fdata()
    encoded = nib.Nifti1Image(((original + 2000) / 2).astype(np.int16), source.affine, source.header)
    encoded.header.set_slope_inter(2, -2000)
    nib.save(encoded, case.image)
    image, _, metadata = read_volume(case, PreprocessConfig(size_hw=None))
    expected = np.clip((original.transpose(2, 1, 0) + 1000) / 1400, 0, 1)
    np.testing.assert_allclose(image, expected, atol=1e-6)
    assert metadata["nifti_scaling"] == {"slope": 2.0, "intercept": -2000.0}
    normalized = write_case("normalized", ct=np.ones((8, 6, 13), dtype=np.float32) * 0.25)
    image, _, _ = read_volume(normalized, PreprocessConfig(intensity_encoding="normalized_0_1"))
    assert torch.all(image == 0.25)


def test_unknown_units_explicit_override(write_case):
    case = write_case(unit="unknown")
    with pytest.raises(ValueError, match="Spatial unit"):
        read_volume(case, PreprocessConfig())
    _, _, metadata = read_volume(case, PreprocessConfig(assume_mm=True))
    assert metadata["assume_mm"]


def test_init_headers_only_and_single_volume_cache(write_case, monkeypatch):
    import ct_pipeline.dataset as module
    cases = [write_case("a"), write_case("b")]
    with monkeypatch.context() as context:
        def fail(*args, **kwargs):
            raise AssertionError("Initialization must not decode a volume")
        context.setattr(nib.arrayproxy.ArrayProxy, "__array__", fail)
        dataset = CTSlabDataset(cases)
    original, calls = module.read_volume, []
    def tracked(case, config):
        calls.append(case.case_id)
        return original(case, config)
    monkeypatch.setattr(module, "read_volume", tracked)
    dataset[0]
    dataset[1]
    dataset[13]
    assert calls == ["a", "b"]
    assert dataset.__getstate__()["_cached"] is None


def test_windows_worker_and_seed(write_case):
    dataset = CTSlabDataset([write_case()], PreprocessConfig(size_hw=(8, 8)))
    first = next(iter(make_dataloader(dataset, batch_size=2, num_workers=1)))
    assert first["image_slab"].shape == (2, 1, 11, 8, 8)
    a = next(iter(make_dataloader(dataset, batch_size=2, shuffle=True, seed=3)))
    b = next(iter(make_dataloader(dataset, batch_size=2, shuffle=True, seed=3)))
    assert torch.equal(a["master_slice_index"], b["master_slice_index"])
    assert torch.equal(a["image_slab"], b["image_slab"])


def test_split_dataset_isolation(write_case, tmp_path):
    cases = [write_case(f"c{i}") for i in range(7)]
    records = [{"id": c.case_id, "image": c.image.name, "mask": c.mask.name} for c in cases]
    manifest, path = tmp_path / "cases.json", tmp_path / "splits.json"
    manifest.write_text(json.dumps(records))
    loaded, identity = load_cases(manifest)
    split = create_splits(loaded, identity)
    path.write_text(json.dumps(split))
    train = CTSlabDataset.from_split(manifest, path, 0, "train")
    test = CTSlabDataset.from_split(manifest, path, 0, "test")
    assert not {c.case_id for c in train.cases} & {c.case_id for c in test.cases}
    with pytest.raises(ValueError, match="preserve"):
        make_dataloader(test, shuffle=True)
