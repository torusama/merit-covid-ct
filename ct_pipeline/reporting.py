"""Descriptive primary-dataset reports from an existing audit and native masks."""
from __future__ import annotations

from collections import Counter
import json

import nibabel as nib
import numpy as np

from .config import PreprocessConfig
from .geometry import affine_mm, inspect_pair


def positive_runs(flags):
    runs, length = [], 0
    for flag in flags:
        if flag:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return runs


def mask_statistics(mask_xyz, canonical_affine_mm):
    """Native canonical grid only: no resize, clipping, padding or Z interpolation."""
    mask = np.asarray(mask_xyz)
    if mask.ndim != 3 or not mask.size or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected a nonempty finite 3D binary mask")
    affine = np.asarray(canonical_affine_mm, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("Expected a finite invertible affine in mm")
    mask = mask.astype(bool, copy=False)
    areas = np.count_nonzero(mask, axis=(0, 1)).astype(np.int64)
    positive = areas > 0
    indices = np.flatnonzero(positive)
    runs = positive_runs(positive)
    adjacent = []
    for z in range(1, mask.shape[2]):
        if positive[z - 1] and positive[z]:
            intersection = int(np.count_nonzero(mask[:, :, z - 1] & mask[:, :, z]))
            adjacent.append(2 * intersection / int(areas[z - 1] + areas[z]))
    voxel_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    pixel_mm2 = float(np.linalg.norm(np.cross(affine[:3, 0], affine[:3, 1])))
    count = int(areas.sum())
    return {
        "slices": int(len(areas)), "positive_slices": int(positive.sum()),
        "empty_slices": int((~positive).sum()), "positive_slice_fraction": float(positive.mean()),
        "first_positive_slice": int(indices[0]) if len(indices) else None,
        "last_positive_slice": int(indices[-1]) if len(indices) else None,
        "positive_run_count": len(runs), "max_positive_run_length": max(runs, default=0),
        "positive_run_lengths": runs, "lesion_voxels": count, "total_voxels": int(mask.size),
        "lesion_fraction_volume": count / mask.size, "voxel_volume_mm3": voxel_mm3,
        "lesion_volume_ml": count * voxel_mm3 / 1000,
        "pixel_area_mm2": pixel_mm2, "slice_center_step_mm": float(np.linalg.norm(affine[:3, 2])),
        "adjacent_positive_pairs": len(adjacent),
        "adjacent_gt_dice_median": float(np.median(adjacent)) if adjacent else None,
        "areas_by_slice_voxels": areas.tolist(),
    }


def describe_case(case, audit_record, assume_mm=False):
    """Recheck current headers and complete mask; reuse recorded CT intensity audit.

    A mask-count mismatch or changed geometry invalidates reuse. This does not
    establish byte identity of the CT or replace a fresh complete intensity audit.
    """
    if audit_record["case_id"] != case.case_id or audit_record["group_id"] != case.group_id:
        raise ValueError(f"{case.case_id}: audit identity differs from manifest")
    if audit_record["z_resampled"]:
        raise ValueError("Reporting requires an audit without Z resampling")
    ct, mask = inspect_pair(case, PreprocessConfig(assume_mm=assume_mm))
    if list(ct.shape) != audit_record["original_shape_xyz"] or not np.allclose(
            ct.affine, audit_record["original_affine"], atol=1e-4, rtol=0):
        raise ValueError(f"{case.case_id}: current geometry differs from saved audit")
    unit = ct.header.get_xyzt_units()[0]
    if unit != audit_record["spatial_unit_in_header"]:
        raise ValueError(f"{case.case_id}: current spatial units differ from saved audit")
    canonical = nib.as_closest_canonical(mask)
    canonical_mm = affine_mm(canonical, assume_mm)
    if list(canonical.shape) != audit_record["canonical_shape_xyz"] or not np.allclose(
            canonical_mm, audit_record["canonical_affine_mm"], atol=1e-4, rtol=0):
        raise ValueError(f"{case.case_id}: current canonical geometry differs from saved audit")
    labels = np.asarray(canonical.dataobj)
    stats = mask_statistics(labels, canonical_mm)
    if stats["lesion_voxels"] != audit_record["native_lesion_voxels"] or np.unique(labels).astype(int).tolist() != audit_record["label_values"]:
        raise ValueError(f"{case.case_id}: current mask differs from saved audit statistics")
    spacing = nib.affines.voxel_sizes(affine_mm(ct, assume_mm))
    scalar = {key: value for key, value in stats.items() if key != "areas_by_slice_voxels"}
    for key, value in list(scalar.items()):
        if isinstance(value, list):
            scalar[key] = json.dumps(value, separators=(",", ":"))
    row = {
        "case_id": case.case_id, "group_id": case.group_id,
        "original_shape_x": int(ct.shape[0]), "original_shape_y": int(ct.shape[1]), "original_shape_z": int(ct.shape[2]),
        "original_orientation": "".join(nib.aff2axcodes(ct.affine)), "canonical_orientation": "RAS",
        "spacing_x_mm": float(spacing[0]), "spacing_y_mm": float(spacing[1]), "spacing_z_mm": float(spacing[2]),
        "spatial_unit_in_header": unit, "spatial_units_assumed_mm": unit == "unknown",
        "ct_dtype": str(ct.get_data_dtype()), "mask_dtype": str(mask.get_data_dtype()),
        "ct_min": audit_record["scaled_intensity_range"][0], "ct_max": audit_record["scaled_intensity_range"][1],
        "ct_scaling_slope": float(ct.dataobj.slope), "ct_scaling_intercept": float(ct.dataobj.inter),
        "mask_label_values": json.dumps(audit_record["label_values"]),
        "geometry_match": True, "mask_finite_binary": True, "z_resampled": False,
        **scalar,
    }
    # Retain every original audit field, including affine matrices, in CSV cells.
    # Nested objects use JSON within a correctly quoted CSV field.
    row.update({f"audit_{key}": json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                for key, value in audit_record.items() if key not in ("case_id", "group_id")})
    trajectory = {"case_id": case.case_id, "areas_by_slice_voxels": stats["areas_by_slice_voxels"],
                  "pixel_area_mm2": stats["pixel_area_mm2"], "slice_center_step_mm": stats["slice_center_step_mm"],
                  "spatial_units_assumed_mm": unit == "unknown"}
    return row, trajectory


def summarize(rows, trajectories, identity, audit_config):
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty rows with unique case IDs")
    if {r["case_id"] for r in rows} != {t["case_id"] for t in trajectories} or len(rows) != len(trajectories):
        raise ValueError("Trajectory/case coverage mismatch")
    slices = sum(r["slices"] for r in rows)
    total = sum(r["total_voxels"] for r in rows)
    lesion = sum(r["lesion_voxels"] for r in rows)
    fractions = [r["lesion_fraction_volume"] for r in rows]
    return {
        "cases": len(rows), "slices": slices,
        "positive_slices": sum(r["positive_slices"] for r in rows),
        "empty_slices": sum(r["empty_slices"] for r in rows),
        "total_voxels": total, "lesion_voxels": lesion,
        "lesion_fraction_volume_pooled": lesion / total,
        "lesion_fraction_volume_macro": float(np.mean(fractions)),
        "lesion_volume_ml_total": sum(r["lesion_volume_ml"] for r in rows),
        "slice_count_min": min(r["slices"] for r in rows), "slice_count_max": max(r["slices"] for r in rows),
        "slice_count_median": float(np.median([r["slices"] for r in rows])),
        "shape_counts": dict(Counter(f"{r['original_shape_x']}x{r['original_shape_y']}x{r['original_shape_z']}" for r in rows)),
        "spacing_z_mm_counts": dict(Counter(f"{r['spacing_z_mm']:.6g}" for r in rows)),
        "orientation_counts": dict(Counter(r["original_orientation"] for r in rows)),
        "header_spatial_unit_counts": dict(Counter(r["spatial_unit_in_header"] for r in rows)),
        "cases_using_assumed_mm": [r["case_id"] for r in rows if r["spatial_units_assumed_mm"]],
        "ct_scaled_intensity_range": [min(r["ct_min"] for r in rows), max(r["ct_max"] for r in rows)],
        "all_geometry_match": all(r["geometry_match"] for r in rows),
        "all_masks_finite_binary": all(r["mask_finite_binary"] for r in rows),
        "identity": identity, "source_audit_config": audit_config,
        "slice_order": "native-resolution canonical RAS, zero-based axial index; orientation only, no resize or Z resampling",
        "physical_units_note": "Physical measurements for cases with unknown header units depend on the explicit assume-mm option.",
        "ct_intensity_source": "Complete CT intensity/finite checks reused from saved audit; current CT headers and masks rechecked. CT bytes are not checksummed.",
        "lesion_denominator": "all voxels in the complete CT volume, including air outside the body; not lung volume",
        "adjacent_gt_dice_note": "Similarity between neighboring ground-truth masks when both contain lesion; not model performance.",
        "unavailable_metrics": {"lesion_fraction_lung": "No lung mask supplied", "lesion_outside_lung": "No lung mask supplied"},
        "per_case_slice_areas": trajectories,
    }
