"""Foreground volume metrics, complete slice assembly and explicit group macro means."""
from __future__ import annotations

from collections import defaultdict

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree

from .config import PreprocessConfig
from .geometry import restore_volume
from .preprocessing import read_volume


def as_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def probabilities(value):
    array = as_array(value)
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError("Predictions must be finite probabilities in [0, 1], not logits")
    return array


def binary_ground_truth(value):
    array = as_array(value)
    if not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError("Ground truth must contain only finite 0/1 labels")
    return array.astype(bool)


def validate_threshold(threshold):
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Threshold must be strictly between 0 and 1")


def hd95_mm(pred, truth, affine):
    """95th percentile of pooled bidirectional 6-connected surface-center distances.

    Full affine world coordinates support anisotropy, obliquity and shear.
    Undefined when either surface is empty; never silently reported as 0 mm.
    """
    if not pred.any() or not truth.any():
        return None
    structure = generate_binary_structure(3, 1)
    pred_surface = pred & ~binary_erosion(pred, structure=structure, border_value=0)
    gt_surface = truth & ~binary_erosion(truth, structure=structure, border_value=0)
    p = nib.affines.apply_affine(affine, np.argwhere(pred_surface))
    g = nib.affines.apply_affine(affine, np.argwhere(gt_surface))
    distances = np.concatenate([cKDTree(g).query(p)[0], cKDTree(p).query(g)[0]])
    return float(np.percentile(distances, 95))


def volume_metrics(prediction, ground_truth, threshold=0.5, affine_mm=None, include_hd95=False):
    """Arrays are 3D X,Y,Z on the same native grid; probabilities use >= threshold."""
    validate_threshold(threshold)
    pred, truth = probabilities(prediction), binary_ground_truth(ground_truth)
    if pred.shape != truth.shape or truth.ndim != 3 or not truth.size:
        raise ValueError(f"Expected matching nonempty 3D shapes, got {pred.shape} and {truth.shape}")
    pred = pred >= threshold
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    both_empty = not pred.any() and not truth.any()
    result = {
        "dice": 2 * tp / (2 * tp + fp + fn) if not both_empty else 1.0,
        "iou": tp / (tp + fp + fn) if not both_empty else 1.0,
        "precision": tp / (tp + fp) if tp + fp else float(both_empty),
        "recall": tp / (tp + fn) if tp + fn else float(both_empty),
        "tp": tp, "fp": fp, "fn": fn,
        "prediction_foreground_voxels": tp + fp, "ground_truth_foreground_voxels": tp + fn,
    }
    if affine_mm is not None:
        affine = np.asarray(affine_mm, dtype=float)
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError("A finite invertible 4x4 affine in millimetres is required")
        voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000
        signed_error = (fp - fn) * voxel_ml
        result.update({
            "prediction_lesion_ml": (tp + fp) * voxel_ml,
            "ground_truth_lesion_ml": (tp + fn) * voxel_ml,
            "lesion_volume_error_ml": signed_error,
            "absolute_lesion_volume_error_ml": abs(signed_error),
            "relative_lesion_volume_error": abs(fp - fn) / (tp + fn) if tp + fn else None,
        })
        if include_hd95:
            result["hd95_mm"] = hd95_mm(pred, truth, affine)
            result["hd95_status"] = "defined" if pred.any() and truth.any() else "undefined_empty_mask"
    elif include_hd95:
        raise ValueError("HD95 requires affine_mm")
    return result


class SliceReconstructor:
    """Reject duplicate, missing, out-of-range, non-finite and mismatched slices."""
    def __init__(self, shape_zyx):
        if len(shape_zyx) != 3 or any(type(n) is not int or n <= 0 for n in shape_zyx):
            raise ValueError("Expected a positive integer Z,Y,X shape")
        self.volume = np.empty(shape_zyx, dtype=np.float32)
        self.seen = np.zeros(shape_zyx[0], dtype=bool)

    def add(self, master_slice_index, prediction):
        i = master_slice_index
        if not isinstance(i, (int, np.integer)) or not 0 <= i < len(self.seen):
            raise ValueError(f"Invalid master slice index: {i}")
        if self.seen[i]:
            raise ValueError(f"Duplicate prediction for master slice {i}")
        pred = probabilities(prediction)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.shape != self.volume.shape[1:]:
            raise ValueError(f"Slice shape mismatch: {pred.shape} vs {self.volume.shape[1:]}")
        self.volume[i] = pred
        self.seen[i] = True

    def finish(self):
        if not self.seen.all():
            raise ValueError(f"Missing master slices: {np.flatnonzero(~self.seen).tolist()}")
        return self.volume.copy()


METRICS = ("dice", "iou", "precision", "recall", "hd95_mm", "lesion_volume_error_ml",
           "absolute_lesion_volume_error_ml", "relative_lesion_volume_error")


def aggregate_results(rows, identity=None):
    """Mean volumes within each group, then equal-weight macro mean across groups.

    Multiple series do not give a patient more weight. For unknown patient IDs
    these remain group/volume macro statistics, explicitly not verified patient means.
    """
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty results with unique case IDs")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    group_rows = []
    for group, items in sorted(grouped.items()):
        entry = {"group_id": group, "case_ids": sorted(r["case_id"] for r in items)}
        entry["valid_case_counts"] = {}
        for metric in METRICS:
            values = [r[metric] for r in items if r.get(metric) is not None]
            if any(not np.isfinite(v) for v in values):
                raise ValueError(f"Non-finite {metric}")
            entry[metric] = float(np.mean(values)) if values else None
            entry["valid_case_counts"][metric] = len(values)
        group_rows.append(entry)
    summary = {}
    for metric in METRICS:
        values = [g[metric] for g in group_rows if g[metric] is not None]
        summary[metric] = {"mean": float(np.mean(values)) if values else None,
                           "std": float(np.std(values, ddof=0)) if values else None,
                           "valid_groups": len(values), "undefined_groups": len(group_rows) - len(values)}
    return {"per_case": rows, "per_group": group_rows, "macro": summary,
            "identity": identity or {"patient_ids_available": False, "identity_note": "Patient identity unverified"},
            "aggregation": "mean per-volume metrics within group, then equal-weight group mean; population std (ddof=0)",
            "undefined_metrics": "null excluded from the metric mean; valid/undefined counts reported",
            "empty_policy": "both empty: overlap/precision/recall=1; exactly one empty: all four=0; HD95=null if either empty"}


class VolumeEvaluator:
    """Evaluate one case at a time and retain scalar results only.

    Slice indices are canonical RAS indices emitted by CTSlabDataset. Predictions
    are restored as probabilities to the original grid BEFORE thresholding.
    """
    def __init__(self, config=None, threshold=0.5, include_hd95=False, identity=None):
        validate_threshold(threshold)
        self.config = config or PreprocessConfig()
        self.threshold, self.include_hd95, self.identity = threshold, include_hd95, identity
        self.rows = []

    def evaluate_case(self, case, slice_predictions):
        if any(row["case_id"] == case.case_id for row in self.rows):
            raise ValueError(f"Case already evaluated: {case.case_id}")
        image, _, metadata = read_volume(case, self.config)
        shape = tuple(image.shape)
        del image
        reconstruction = SliceReconstructor(shape)
        for index, prediction in slice_predictions:
            reconstruction.add(index, prediction)
        native = restore_volume(reconstruction.finish(), metadata, probabilities=True)
        gt = nib.load(case.mask).get_fdata(dtype=np.float32, caching="unchanged")
        result = volume_metrics(native, gt, self.threshold, metadata["original_affine_mm"], self.include_hd95)
        row = {"case_id": case.case_id, "group_id": case.group_id, **result}
        self.rows.append(row)
        return row

    def report(self):
        return {**aggregate_results(self.rows, self.identity), "threshold": self.threshold,
                "evaluation_grid": "native NIfTI X,Y,Z; full-volume foreground metrics"}
