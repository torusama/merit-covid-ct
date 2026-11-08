"""Preflight math and train-only access checks; synthetic NIfTI fixtures."""

import nibabel as nib
import numpy as np
import pytest
import yaml

from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.preflight import (
    IntensityAccumulator, resize_plane_audit, native_blocks, audit_scope, inspect_training,
)
from merit_covid.data.pipeline.preprocessing import read_arrays
from merit_covid.data.pipeline.config import PreprocessConfig


def test_streaming_moments_and_integer_percentiles_are_exact():
    values = np.array([-2048, -1000, -20, 0, 2, 100, 400, 32000], np.float32)
    accumulator = IntensityAccumulator()
    for block in np.array_split(values, 3):
        accumulator.update(block)
    result = accumulator.result()
    assert result["mean"] == pytest.approx(values.mean(dtype=np.float64))
    assert result["std_population"] == pytest.approx(values.std(dtype=np.float64))
    assert result["p50"] == np.percentile(values, 50)
    assert result["p99"] == pytest.approx(np.percentile(values, 99))
    assert accumulator.histogram.sum() == len(values)
    with pytest.raises(ValueError):
        accumulator.update([np.nan])


def test_histogram_explicit_overflow_and_fractional_values():
    accumulator = IntensityAccumulator()
    accumulator.update(np.array([-40000, 0.25, 0.75, 65536], np.float32))
    result = accumulator.result()
    assert result["histogram_underflow"] == result["histogram_overflow"] == 1
    assert result["noninteger_voxels"] == 2
    assert accumulator.histogram.sum() == 2
    assert result["p0_1"] is None and result["p99_9"] is None


def test_resize_loss_is_measured_on_native_support_not_unequal_counts():
    full = np.ones((8, 8), bool)
    stats, down, up = resize_plane_audit(full, 4)
    assert down.sum() == 16 and full.sum() == 64
    assert stats["native_lesion_voxels_lost_roundtrip"] == 0
    assert stats["lost_components_2d"] == 0 and up.all()
    small = np.zeros((8, 8), bool)
    small[0, 0] = True  # Not sampled by 8 -> 4 nearest-exact.
    stats, down, up = resize_plane_audit(small, 4)
    assert stats["lost_components_2d"] == stats["lesion_voxels_in_lost_components"] == 1
    assert stats["positive_to_empty_slices"] == stats["native_lesion_voxels_lost_roundtrip"] == 1
    assert not down.any() and not up.any()


def test_resize_matches_actual_torch_nearest_exact():
    torch = pytest.importorskip("torch")
    plane = np.random.default_rng(7).integers(0, 2, (13, 17)).astype(bool)
    for size in (4, 8, 32):
        _, down, up = resize_plane_audit(plane, size)
        tensor = torch.from_numpy(plane.astype(np.float32))[None, None]
        expected = torch.nn.functional.interpolate(tensor, size=(size, size), mode="nearest-exact")
        restored = torch.nn.functional.interpolate(expected, size=plane.shape, mode="nearest-exact")
        np.testing.assert_array_equal(down, expected[0, 0].numpy())
        np.testing.assert_array_equal(up, restored[0, 0].numpy())


@pytest.mark.parametrize("sign", [1, -1])
def test_streaming_matches_shared_reader_orientation_and_scaling(nifti_case, sign):
    source, _, _ = nifti_case(shape=(9, 11, 5), affine=np.diag([-.7, -.8, sign*3., 1]))
    case = Case(source.case_id, source.image, source.infection_mask, source.case_id)
    image, mask, _ = read_arrays(case, PreprocessConfig())
    assembled_image, assembled_mask = np.zeros_like(image), np.zeros_like(mask)
    for indices, ct, gt in native_blocks(case, block_depth=2):
        assembled_image[list(indices)] = ct
        assembled_mask[list(indices)] = gt
    np.testing.assert_array_equal(image, assembled_image)
    np.testing.assert_array_equal(mask, assembled_mask)


def test_audit_reads_only_fold_training_voxels(tmp_path, monkeypatch):
    from merit_covid.engine.synthetic import synthetic_experiment
    config = synthetic_experiment(tmp_path / "data")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _, train, cases, split, provenance = audit_scope(path)
    protected = {str(c.image): str(c.mask) for c in cases if c not in train}
    protected_paths = set(protected) | set(protected.values())
    original_load = nib.load
    def guarded_load(filename, *args, **kwargs):
        assert str(filename) not in protected_paths, "Validation/test voxel/header access"
        return original_load(filename, *args, **kwargs)
    monkeypatch.setattr(nib, "load", guarded_load)
    result = inspect_training(path, tmp_path / "audit", block_depth=1)
    assert result["status"] == "PASS" and result["audited_cases"] == len(train)
    assert result["source_files_modified"] == []
    assert all(r["group_overlap"] == 0 for r in result["provenance"]["overlaps"])
    assert (tmp_path / "audit/resize_cases.csv").exists()
    assert (tmp_path / "audit/intensity_histogram.png").exists()
    with pytest.raises(FileExistsError):
        inspect_training(path, tmp_path / "audit")


def test_complete_compact_pilot_and_profile(tmp_path):
    from merit_covid.engine.synthetic import synthetic_experiment
    from merit_covid.engine.preflight import pilot_case, profile_model
    config = synthetic_experiment(tmp_path / "data")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _, train, _, _, _ = audit_scope(path)
    report = pilot_case(path, tmp_path / "pilot", case_id=train[0].case_id, size=32, steps=2, device="cpu")
    assert report["status"] == "PASS", report
    assert report["train"]["optimizer_steps"] == 2
    assert report["weights_changed"] and report["final_gradients_finite"]
    assert report["checkpoint"]["model_exact_match"]
    assert report["checkpoint"]["optimizer_step_counters_match"]
    assert report["inference"]["ct_only_guard"] == "PASS"
    assert report["inference"]["original_shape_xyz"] == [16, 18, 2]
    profile = profile_model(path, tmp_path / "profile.json", size=32, batch_size=2, device="cpu", warmup=1, repeats=1)
    assert profile["status"] == "PASS", profile
    assert profile["optimizer_steps"] == 2 and len(profile["timings_seconds"]) == 1
    assert profile["peak_allocated_mib"] is None
    with pytest.raises(FileExistsError):
        profile_model(path, tmp_path / "profile.json", size=32, batch_size=2, device="cpu")


def test_profile_reports_allocation_failure_without_claiming_fit(tmp_path, monkeypatch):
    import torch
    from merit_covid.engine.synthetic import synthetic_experiment
    import merit_covid.engine.preflight as module
    config = synthetic_experiment(tmp_path / "data")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    def out_of_memory(**kwargs):
        raise torch.OutOfMemoryError("simulated allocation failure")
    monkeypatch.setattr(module, "MERITUNETRPP", out_of_memory)
    report = module.profile_model(path, tmp_path / "failure.json", size=32, batch_size=1, device="cpu")
    assert report["status"] == "OOM" and report["optimizer_steps"] == 0
    assert "mean_step_seconds" not in report
