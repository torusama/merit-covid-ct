"""Strict binary foreground metrics and equal-weight patient aggregation."""

import numpy as np

from merit_covid.data.nifti_io import binary_mask


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Prediction and target must have equal, non-empty shapes")
    pred = binary_mask(np.asarray(prediction)).astype(bool)
    gt = binary_mask(np.asarray(target)).astype(bool)
    tp = int((pred & gt).sum())
    p, g = int(pred.sum()), int(gt.sum())
    union = p + g - tp
    # Sensitivity is 1 when GT is absent, precision 1 when prediction is absent.
    # Dice/IoU still penalize foreground mismatch in either direction.
    return {"dice": 2 * tp / (p + g) if p + g else 1.0,
            "iou": tp / union if union else 1.0,
            "sensitivity": tp / g if g else 1.0,
            "precision": tp / p if p else 1.0}


def aggregate_patients(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """One entry per patient (or volume proxy); never average slice scores here."""
    if not scores:
        raise ValueError("No patient scores supplied")
    keys = {"dice", "iou", "sensitivity", "precision"}
    if any(set(row) != keys or any(not np.isfinite(v) or not 0 <= v <= 1 for v in row.values()) for row in scores.values()):
        raise ValueError("Every patient needs four finite metrics in [0,1]")
    return {key: float(np.mean([row[key] for row in scores.values()])) for key in sorted(keys)}


def evaluate_patients(predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray],
                      patient_ids: dict[str, str | None] | None = None) -> dict:
    """Pool voxel counts over linked volumes, then macro-average patients.

    Keys are volume case IDs. Null/absent verified patient IDs use volume proxies.
    Arrays must already be restored to the original CT grid and binarized.
    """
    if not predictions or predictions.keys() != targets.keys():
        raise ValueError("Prediction and target volume IDs must match exactly")
    if patient_ids is not None and set(patient_ids) != set(predictions):
        raise ValueError("Patient mapping must cover exactly the evaluated volumes")
    counts = {}
    for case_id, prediction in predictions.items():
        target = targets[case_id]
        if prediction.shape != target.shape or prediction.ndim != 3 or prediction.size == 0:
            raise ValueError("Patient evaluation requires aligned original-grid 3D volumes")
        pred, gt = binary_mask(prediction).astype(bool), binary_mask(target).astype(bool)
        patient = patient_ids.get(case_id) if patient_ids is not None else None
        if patient is not None and (not isinstance(patient, str) or not patient.strip()):
            raise ValueError("Verified patient IDs must be non-empty strings")
        group = f"patient:{patient}" if patient is not None else f"volume:{case_id}"
        totals = counts.setdefault(group, [0, 0, 0])
        for i, value in enumerate((int((pred & gt).sum()), int(pred.sum()), int(gt.sum()))):
            totals[i] += value
    scores = {}
    for group, (tp, p, g) in counts.items():
        scores[group] = {"dice": 2 * tp / (p + g) if p + g else 1.0,
                         "iou": tp / (p + g - tp) if p + g - tp else 1.0,
                         "sensitivity": tp / g if g else 1.0,
                         "precision": tp / p if p else 1.0}
    return {"per_patient": scores, "macro": aggregate_patients(scores)}
