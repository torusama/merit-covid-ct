"""Strict paired NIfTI reading without changing source files."""

from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from .manifest import Case


@dataclass
class Volume:
    image: np.ndarray
    mask: np.ndarray | None
    original_affine: np.ndarray
    affine: np.ndarray
    original_shape: tuple[int, int, int]
    original_orientation: tuple[str, str, str]
    orientation_transform: np.ndarray
    original_header: nib.Nifti1Header
    spatial_unit: str
    mm_per_unit: float
    metadata: dict = field(default_factory=dict)

    @property
    def spacing_mm(self):
        return nib.affines.voxel_sizes(self.affine) * self.mm_per_unit


def verify_alignment(image, mask) -> None:
    if len(image.shape) != 3 or image.shape != mask.shape:
        raise ValueError(f"CT/mask shape mismatch: {image.shape} vs {mask.shape}")
    if not np.allclose(image.affine, mask.affine, rtol=0, atol=1e-4):
        raise ValueError("CT/mask affine mismatch")
    if image.header.get_xyzt_units()[0] != mask.header.get_xyzt_units()[0]:
        raise ValueError("CT/mask spatial unit mismatch")
    if not np.allclose(image.header.get_zooms()[:3], mask.header.get_zooms()[:3], rtol=1e-5, atol=1e-5):
        raise ValueError("CT/mask header spacing mismatch")


def binary_mask(values: np.ndarray, lesion_labels: tuple[int, ...] | None = None) -> np.ndarray:
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError("Mask must contain finite integer labels")
    labels = set(np.unique(values).tolist())
    if lesion_labels is None:
        if not labels <= {0, 1}:
            raise ValueError(f"Expected binary infection mask, found {sorted(labels)}; explicit lesion_labels required")
        return values.astype(np.uint8)
    if not lesion_labels or 0 in lesion_labels or any(type(v) is not int or v < 1 for v in lesion_labels):
        raise ValueError("lesion_labels must contain positive integers")
    return np.isin(values, lesion_labels).astype(np.uint8)


def validate_image_header(case: Case, image, *, assumed_unit: str | None = None) -> tuple[str, float]:
    """Check source geometry against manifest metadata without reading voxels."""
    if len(image.shape) != 3:
        raise ValueError(f"{case.case_id}: expected a 3D CT")
    affine = image.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError(f"{case.case_id}: invalid affine")
    unit = image.header.get_xyzt_units()[0]
    effective_unit = assumed_unit if unit == "unknown" else unit
    if effective_unit not in ("mm", "meter", "micron"):
        raise ValueError(f"{case.case_id}: unknown physical units; supply an evidence-backed assumed_unit")
    scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}[effective_unit]
    if case.shape is not None and tuple(image.shape) != case.shape:
        raise ValueError(f"{case.case_id}: stale manifest shape")
    if case.spacing_mm is not None and not np.allclose(nib.affines.voxel_sizes(affine) * scale, case.spacing_mm, rtol=1e-4, atol=1e-4):
        raise ValueError(f"{case.case_id}: stale manifest spacing")
    if case.affine is not None and not np.allclose(affine, case.affine, rtol=0, atol=1e-4):
        raise ValueError(f"{case.case_id}: stale manifest affine")
    if case.orientation is not None and tuple(nib.aff2axcodes(affine)) != case.orientation:
        raise ValueError(f"{case.case_id}: stale manifest orientation")
    return unit, scale


def load_case(case: Case, *, require_mask=True, lesion_labels=None, assumed_unit=None):
    """Read through Duong LPS/native core; inference can omit all GT paths."""
    from .pipeline.cases import Case as PipelineCase
    from .pipeline.config import PreprocessConfig
    from .pipeline.preprocessing import read_arrays
    image=nib.load(str(case.image))
    unit,scale=validate_image_header(case,image,assumed_unit=assumed_unit)
    if unit=='unknown' and assumed_unit!='mm':
        raise ValueError("Only the agreed assumed_unit='mm' override is supported")
    if require_mask and case.infection_mask is not None:
        verify_alignment(image,nib.load(str(case.infection_mask)))
    if lesion_labels is not None:
        raise ValueError('load_case requires binary masks; map combined labels before loading')
    source=PipelineCase(case.case_id,case.image,case.infection_mask,case.patient_id or case.group_id or case.case_id)
    values,labels,metadata=read_arrays(source,PreprocessConfig(assume_mm=assumed_unit=='mm'),require_mask=require_mask)
    return Volume(values.transpose(2,1,0).copy(),labels.transpose(2,1,0).astype(np.uint8) if labels is not None else None,
                  image.affine.copy(),np.asarray(metadata['canonical_affine_mm'])/np.array([[scale],[scale],[scale],[1]]),
                  tuple(image.shape),tuple(nib.aff2axcodes(image.affine)),
                  nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),nib.orientations.axcodes2ornt(('L','P','S'))),
                  image.header.copy(),unit,scale,metadata)


def save_prediction(prediction: np.ndarray, volume: Volume, path: str | Path) -> None:
    """Accept a binary prediction already restored to the original voxel grid."""
    if prediction.shape != volume.original_shape:
        raise ValueError("Prediction must match original CT shape")
    prediction = binary_mask(prediction)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    header = volume.original_header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)
    result = nib.Nifti1Image(prediction, volume.original_affine, header=header)
    # Preserve independently stored qform/sform and codes from the source header.
    result.set_qform(volume.original_header.get_qform(), int(volume.original_header["qform_code"]))
    result.set_sform(volume.original_header.get_sform(), int(volume.original_header["sform_code"]))
    nib.save(result, str(path))
