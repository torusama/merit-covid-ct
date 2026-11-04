import copy

import nibabel as nib
import numpy as np
import pytest

from ct_pipeline.dataset import PreprocessConfig, read_volume
from ct_pipeline.reporting import describe_case, mask_statistics, summarize


def test_native_mask_statistics_and_physical_units():
    mask = np.zeros((2, 2, 5), dtype=np.uint8)
    mask[0, 0, :2] = 1
    mask[:, 1, 3] = 1
    stats = mask_statistics(mask, np.diag([2.0, 3.0, 5.0, 1.0]))
    assert stats["areas_by_slice_voxels"] == [1, 1, 0, 2, 0]
    assert stats["positive_slices"] == 3 and stats["empty_slices"] == 2
    assert stats["positive_run_lengths"] == [2, 1]
    assert stats["max_positive_run_length"] == 2
    assert stats["first_positive_slice"] == 0 and stats["last_positive_slice"] == 3
    assert stats["lesion_voxels"] == 4 and stats["total_voxels"] == 20
    assert stats["lesion_fraction_volume"] == 0.2
    assert stats["pixel_area_mm2"] == 6
    assert stats["lesion_volume_ml"] == pytest.approx(0.12)
    assert stats["adjacent_positive_pairs"] == 1 and stats["adjacent_gt_dice_median"] == 1


def test_empty_mask_statistics_remain_defined():
    stats = mask_statistics(np.zeros((2, 2, 3)), np.eye(4))
    assert stats["positive_slices"] == stats["lesion_voxels"] == 0
    assert stats["empty_slices"] == 3
    assert stats["first_positive_slice"] is None
    assert stats["adjacent_gt_dice_median"] is None
    assert stats["positive_run_lengths"] == []


@pytest.mark.parametrize("bad", [np.ones((2, 2)), np.ones((2, 2, 2)) * 2, np.ones((2, 2, 2)) * np.nan])
def test_invalid_masks_rejected(bad):
    with pytest.raises(ValueError):
        mask_statistics(bad, np.eye(4))


def test_report_reuses_audit_but_rechecks_geometry_and_mask(write_case):
    case = write_case(affine=np.diag([-1.0, -2.0, -5.0, 1.0]))
    _, _, metadata = read_volume(case, PreprocessConfig())
    record = {"case_id": case.case_id, "group_id": case.group_id, **metadata}
    row, trajectory = describe_case(case, record)
    assert row["lesion_voxels"] == 4
    assert row["first_positive_slice"] == 11  # Original Z=1 becomes canonical Z=11.
    assert trajectory["areas_by_slice_voxels"][11] == 4
    assert not row["spatial_units_assumed_mm"]
    assert row["audit_original_shape_xyz"] == "[8,6,13]"
    stale = copy.deepcopy(record)
    stale["original_affine"][0][3] += 10
    with pytest.raises(ValueError, match="geometry differs"):
        describe_case(case, stale)
    stale = copy.deepcopy(record)
    stale["native_lesion_voxels"] += 1
    with pytest.raises(ValueError, match="mask differs"):
        describe_case(case, stale)


def test_missing_units_stay_explicit(write_case):
    case = write_case(unit="unknown")
    _, _, metadata = read_volume(case, PreprocessConfig(assume_mm=True))
    record = {"case_id": case.case_id, "group_id": case.group_id, **metadata}
    with pytest.raises(ValueError, match="Spatial unit"):
        describe_case(case, record)
    row, _ = describe_case(case, record, assume_mm=True)
    assert row["spatial_unit_in_header"] == "unknown" and row["spatial_units_assumed_mm"]


def test_summary_distinguishes_pooled_and_macro(write_case):
    rows, trajectories = [], []
    for case in (write_case("a"), write_case("b", shape=(8, 6, 20))):
        _, _, metadata = read_volume(case, PreprocessConfig())
        row, trajectory = describe_case(case, {"case_id": case.case_id, "group_id": case.group_id, **metadata})
        rows.append(row)
        trajectories.append(trajectory)
    summary = summarize(rows, trajectories, {"patient_ids_available": False}, {})
    assert summary["cases"] == 2 and summary["slices"] == 33
    assert summary["lesion_voxels"] == 8
    assert summary["lesion_fraction_volume_pooled"] != summary["lesion_fraction_volume_macro"]
    with pytest.raises(ValueError, match="coverage"):
        summarize(rows, trajectories[:-1], {}, {})
