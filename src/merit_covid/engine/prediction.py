"""Model-independent prediction contract and bridges to the EXISTING evaluator.

No metrics or threshold selection here. Every output is a complete native-size
LPS probability volume, with geometry to restore to the original NIfTI grid.
"""

from copy import deepcopy
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from scipy.special import expit

from merit_covid.data.pipeline.cases import identifier
from merit_covid.data.pipeline.geometry import restore_volume
from merit_covid.data.pipeline.metrics import SliceReconstructor, probabilities, as_array


GEOMETRY_KEYS = (
    "original_shape_xyz", "canonical_shape_xyz", "original_affine", "original_affine_mm",
    "canonical_affine_mm", "canonical_to_original_orientation", "original_orientation",
    "canonical_orientation", "native_size_hw", "processed_size_hw", "z_resampled",
    "spatial_unit_in_header", "assume_mm",
)


def validate_prediction_metadata(metadata, shape_zyx):
    if not isinstance(metadata, dict) or any(key not in metadata for key in GEOMETRY_KEYS):
        raise ValueError("Prediction needs the complete shared-reader geometry metadata")
    xyz = metadata["canonical_shape_xyz"]
    if (len(xyz) != 3 or any(type(n) is not int or n < 1 for n in xyz)
            or tuple(reversed(xyz)) != tuple(shape_zyx)):
        raise ValueError("Prediction does not match the full canonical CT grid")
    if (tuple(metadata["native_size_hw"]) != (xyz[1], xyz[0])
            or tuple(metadata["processed_size_hw"]) != (xyz[1], xyz[0])
            or metadata["z_resampled"] is not False
            or tuple(metadata["canonical_orientation"]) != ("L", "P", "S")):
        raise ValueError("Unified output must be native-size LPS without Z resampling")
    for key in ("original_affine", "original_affine_mm", "canonical_affine_mm"):
        affine = np.asarray(metadata[key], dtype=np.float64)
        if (affine.shape != (4, 4) or not np.isfinite(affine).all()
                or not np.allclose(affine[3], [0, 0, 0, 1])
                or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
            raise ValueError(f"Invalid prediction geometry: {key}")
    if nib.aff2axcodes(np.asarray(metadata["canonical_affine_mm"])) != ("L", "P", "S"):
        raise ValueError("Canonical affine disagrees with LPS orientation")
    original_shape = metadata["original_shape_xyz"]
    transform = np.asarray(metadata["canonical_to_original_orientation"])
    if (len(original_shape) != 3 or any(type(n) is not int or n < 1 for n in original_shape)
            or transform.shape != (3, 2) or not np.isfinite(transform).all()
            or set(transform[:, 0]) != {0, 1, 2} or not np.isin(transform[:, 1], [-1, 1]).all()):
        raise ValueError("Invalid original shape/orientation transform")
    expected = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(np.asarray(metadata["canonical_affine_mm"])),
        nib.orientations.io_orientation(np.asarray(metadata["original_affine_mm"])),
    )
    if not np.array_equal(transform, expected):
        raise ValueError("Restoration transform disagrees with the original affine")
    restored_shape = tuple(np.asarray(xyz)[np.argsort(transform[:, 0].astype(int))])
    if restored_shape != tuple(original_shape):
        raise ValueError("Restoration transform disagrees with original shape")


@dataclass
class VolumePrediction:
    case_id: str
    probability_zyx: np.ndarray
    metadata: dict

    def __post_init__(self):
        identifier(self.case_id, "case_id")
        array = probabilities(self.probability_zyx)
        if array.ndim != 3 or not array.size:
            raise ValueError("Expected nonempty full-volume probabilities [Z,H,W]")
        validate_prediction_metadata(self.metadata, array.shape)
        if self.metadata.get("case_id", self.case_id) != self.case_id:
            raise ValueError("Prediction case_id disagrees with metadata")
        self.probability_zyx = np.ascontiguousarray(array, dtype=np.float32)
        self.metadata = deepcopy(self.metadata)

    def to_original_xyz(self):
        """Reuse geometry restoration, with no thresholding or file writes."""
        return restore_volume(self.probability_zyx, self.metadata, probabilities=True)

    def slice_predictions(self):
        """Yield the exact input accepted by VolumeEvaluator.evaluate_case()."""
        for z, plane in enumerate(self.probability_zyx):
            yield z, plane


def prediction_from_slices(case_id, slice_predictions, metadata, *, policy=None):
    """Bridge 2D/master-slice models to the same full-volume contract."""
    xyz = metadata["canonical_shape_xyz"]
    shape = tuple(reversed(xyz))
    validate_prediction_metadata(metadata, shape)
    assembly = SliceReconstructor(shape)
    for index, plane in slice_predictions:
        assembly.add(index, plane)
    record = deepcopy(metadata)
    record["prediction_policy"] = deepcopy(policy or {})
    return VolumePrediction(case_id, assembly.finish(), record)


def binary_logits_to_probabilities(logits):
    """Output-layer sigmoid only, for logits [B,1,H,W] or [B,1,D,H,W]."""
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu().numpy()
    array = as_array(logits)
    if (array.ndim not in (4, 5) or array.shape[1] != 1 or any(n < 1 for n in array.shape)
            or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all()):
        raise ValueError("Expected finite floating binary logits [B,1,H,W] or [B,1,D,H,W]")
    return np.ascontiguousarray(expit(array[:, 0]), dtype=np.float32)
