"""Strict binary foreground metrics and equal-weight patient aggregation."""
import numpy as np
from merit_covid.data.nifti_io import binary_mask

def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError('Prediction and target must have equal, non-empty shapes')
    pred = binary_mask(np.asarray(prediction)).astype(bool)
    gt = binary_mask(np.asarray(target)).astype(bool)
    from merit_covid.data.pipeline.metrics import volume_metrics
    metrics = volume_metrics(pred.reshape(-1, 1, 1), gt.reshape(-1, 1, 1))
    return {'dice': metrics['dice'], 'iou': metrics['iou'], 'sensitivity': metrics['recall'], 'precision': metrics['precision']}

def aggregate_patients(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """One entry per patient (or volume proxy); never average slice scores here."""
    if not scores:
        raise ValueError('No patient scores supplied')
    keys = {'dice', 'iou', 'sensitivity', 'precision'}
    if any((set(row) != keys or any((not np.isfinite(v) or not 0 <= v <= 1 for v in row.values())) for row in scores.values())):
        raise ValueError('Every patient needs four finite metrics in [0,1]')
    return {key: float(np.mean([row[key] for row in scores.values()])) for key in sorted(keys)}

def evaluate_patients(predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray], patient_ids: dict[str, str | None] | None=None) -> dict:
    """Mean per-volume metrics within linked groups, then equal-weight group macro.

    Keys are volume case IDs. Null/absent verified patient IDs use volume proxies.
    Arrays must already be restored to the original CT grid and binarized.
    """
    if not predictions or predictions.keys() != targets.keys():
        raise ValueError('Prediction and target volume IDs must match exactly')
    if patient_ids is not None and set(patient_ids) != set(predictions):
        raise ValueError('Patient mapping must cover exactly the evaluated volumes')
    from merit_covid.data.pipeline.metrics import volume_metrics, aggregate_results
    rows = []
    for case_id, prediction in predictions.items():
        target = targets[case_id]
        if prediction.shape != target.shape or prediction.ndim != 3 or prediction.size == 0:
            raise ValueError('Patient evaluation requires aligned original-grid 3D volumes')
        pred, gt = (binary_mask(prediction), binary_mask(target))
        patient = patient_ids.get(case_id) if patient_ids is not None else None
        if patient is not None and (not isinstance(patient, str) or not patient.strip()):
            raise ValueError('Verified patient IDs must be non-empty strings')
        group = f'patient:{patient}' if patient is not None else f'volume:{case_id}'
        rows.append({'case_id': case_id, 'group_id': group, **volume_metrics(pred, gt)})
    report = aggregate_results(rows)
    scores = {row['group_id']: {'dice': row['dice'], 'iou': row['iou'], 'sensitivity': row['recall'], 'precision': row['precision']} for row in report['per_group']}
    return {'per_patient': scores, 'macro': aggregate_patients(scores)}
