"""Shape, affine, orientation, đơn vị mm và đường quay về geometry gốc.

File này không window cường độ CT và không lấy slab. inspect_pair() chỉ đọc
headers; restore_volume() được dùng sau khi đã ghép đủ prediction từng lát.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from .cases import Case
from .config import PreprocessConfig


def affine_mm(image, assume_mm: bool = False) -> np.ndarray:
    """Đổi tọa độ affine về mm; từ chối đơn vị unknown theo mặc định."""
    unit = image.header.get_xyzt_units()[0]
    unit_factors = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in unit_factors and not (unit == "unknown" and assume_mm):
        raise ValueError(
            f"Spatial unit is {unit!r}; "
            "use assume_mm only with documented source evidence"
        )

    affine = np.asarray(image.affine, dtype=np.float64).copy()
    affine[:3] *= unit_factors.get(unit, 1.0)
    return affine


def _validate_header(image, case_id: str, image_name: str) -> None:
    affine = image.affine
    if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
        raise ValueError(f"{case_id}: {image_name} must be a nonempty 3D volume")

    if (
        not np.isfinite(affine).all()
        or abs(np.linalg.det(affine[:3, :3])) < 1e-12
        or not np.allclose(affine[3], [0, 0, 0, 1])
    ):
        raise ValueError(f"{case_id}: invalid {image_name} affine")

    spacing = np.asarray(image.header.get_zooms()[:3])
    if (
        not np.isfinite(spacing).all()
        or np.any(spacing <= 0)
        or not np.allclose(
            spacing,
            nib.affines.voxel_sizes(affine),
            atol=1e-4,
            rtol=1e-4,
        )
    ):
        raise ValueError(f"{case_id}: {image_name} spacing disagrees with affine")


def inspect_pair(case: Case, config: PreprocessConfig, *, require_mask: bool = True):
    """Mở headers CT/mask và báo lỗi nếu hai file khác geometry.

    Không giải mã toàn bộ volume tại bước này. Không resize mask để che lệch.
    """
    ct = nib.load(case.image)
    _validate_header(ct, case.case_id, "CT")
    if case.mask is None:
        affine_mm(ct, config.assume_mm)
        if require_mask:
            raise ValueError(f"{case.case_id}: mask required")
        return ct, None
    mask = nib.load(case.mask)
    _validate_header(mask, case.case_id, "mask")

    ct_affine_mm = affine_mm(ct, config.assume_mm)
    mask_affine_mm = affine_mm(mask, config.assume_mm)
    if (
        ct.shape != mask.shape
        or not np.allclose(ct_affine_mm, mask_affine_mm, atol=1e-4, rtol=0)
        or nib.aff2axcodes(ct.affine) != nib.aff2axcodes(mask.affine)
    ):
        raise ValueError(
            f"{case.case_id}: CT/mask geometry mismatch "
            "(shape, affine, orientation or spacing)"
        )

    return ct, mask


def _lps_transform(image) -> np.ndarray:
    """Đổi hướng các trục mảng về LPS; tọa độ world của NiBabel vẫn là RAS+."""
    return nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine),
        nib.orientations.axcodes2ornt(("L", "P", "S")),
    )


def reorient_to_lps(image):
    """Chỉ đảo/hoán vị voxel và cập nhật affine; không nội suy hay sửa file.

    Ảnh đã LPS được trả nguyên trạng. Không làm thẳng ảnh oblique.
    """
    return image.as_reoriented(_lps_transform(image))


def canonical_shape(image) -> tuple[int, int, int]:
    """Tính shape sau thống nhất LPS chỉ từ header, không đọc voxel."""
    transform = _lps_transform(image)
    reordered_shape = np.asarray(image.shape)[np.argsort(transform[:, 0])]
    return tuple(int(size) for size in reordered_shape)


def geometry_metadata(original_ct, canonical_ct, assume_mm: bool) -> dict:
    """Giữ tọa độ native để offsets và prediction có thể quay về ảnh gốc."""
    canonical_affine_mm = affine_mm(canonical_ct, assume_mm)
    inverse_orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(canonical_ct.affine),
        nib.orientations.io_orientation(original_ct.affine),
    )
    return {
        "original_shape_xyz": list(original_ct.shape),
        "canonical_shape_xyz": list(canonical_ct.shape),
        "original_affine": original_ct.affine.tolist(),
        "original_affine_mm": affine_mm(original_ct, assume_mm).tolist(),
        "canonical_affine_mm": canonical_affine_mm.tolist(),
        "canonical_to_original_orientation": inverse_orientation.tolist(),
        "original_orientation": list(nib.aff2axcodes(original_ct.affine)),
        "canonical_orientation": list(nib.aff2axcodes(canonical_ct.affine)),
        "native_spacing_mm": nib.affines.voxel_sizes(canonical_affine_mm).tolist(),
        "spatial_unit_in_header": original_ct.header.get_xyzt_units()[0],
        "assume_mm": assume_mm,
        "z_resampled": False,
    }


def restore_volume(prediction_zyx, metadata: dict, probabilities: bool = False):
    """Prediction Z,Y,X → orientation X,Y,Z gốc; mặc định không nội suy.

    Chỉ resize XY native nếu trước đó caller chủ động yêu cầu resize đầu vào.
    Binary mask dùng nearest neighbor. Probability dùng bilinear, sau đó
    evaluator mới threshold trên native grid. Không nội suy theo chiều Z.
    """
    prediction = np.asarray(prediction_zyx, dtype=np.float32)
    canonical_xyz = metadata["canonical_shape_xyz"]
    expected_shape = (canonical_xyz[2], *metadata["processed_size_hw"])
    if tuple(prediction.shape) != expected_shape or not np.isfinite(prediction).all():
        raise ValueError(
            f"Prediction shape must be {expected_shape}, finite; "
            f"found {tuple(prediction.shape)}"
        )

    if not probabilities and not np.all((prediction == 0) | (prediction == 1)):
        raise ValueError("Binary restoration requires 0/1 predictions")
    if probabilities and not np.all((prediction >= 0) & (prediction <= 1)):
        raise ValueError("Probability restoration requires values in [0, 1]")

    if tuple(prediction.shape[1:]) != tuple(metadata["native_size_hw"]):
        import torch
        from torch.nn import functional as F
        interpolation = (
            {"mode": "bilinear", "align_corners": False}
            if probabilities
            else {"mode": "nearest-exact"}
        )
        prediction = F.interpolate(
            torch.from_numpy(np.ascontiguousarray(prediction))[:, None],
            size=metadata["native_size_hw"],
            **interpolation,
        )[:, 0].numpy()

    canonical_array_xyz = prediction.transpose(2, 1, 0)
    original_array_xyz = nib.orientations.apply_orientation(
        canonical_array_xyz,
        np.asarray(metadata["canonical_to_original_orientation"]),
    )
    if tuple(original_array_xyz.shape) != tuple(metadata["original_shape_xyz"]):
        raise ValueError("Restored prediction shape disagrees with original volume")

    return np.ascontiguousarray(original_array_xyz)
