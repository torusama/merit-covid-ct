"""Một CT/mask gốc → volume đã xử lý. Đọc read_volume() để thấy toàn bộ luồng.

Quy ước đầu vào NIfTI: X,Y,Z. Quy ước tensor trả về: Z,Y,X (Z,H,W).
Các thao tác chỉ tạo mảng/tensor trong RAM; không ghi đè NIfTI gốc.
"""

from __future__ import annotations

import numpy as np

from .cases import Case
from .config import PreprocessConfig
from .geometry import geometry_metadata, inspect_pair, reorient_to_lps


def validate_voxels(image_xyz: np.ndarray, mask_xyz: np.ndarray | None, case_id: str) -> None:
    """Kiểm tra toàn bộ CT/mask sau NIfTI scaling, trước window/resize."""
    if not np.isfinite(image_xyz).all() or (mask_xyz is not None and not np.isfinite(mask_xyz).all()):
        raise ValueError(f"{case_id}: CT/mask contains NaN or Inf")
    if mask_xyz is not None and not np.isin(mask_xyz, [0, 1]).all():
        raise ValueError(
            f"{case_id}: mask labels must be exactly 0/1; "
            f"found {np.unique(mask_xyz)}"
        )


def normalize_ct(
    image_xyz: np.ndarray,
    config: PreprocessConfig,
    case_id: str,
) -> np.ndarray:
    """Mặc định trả nguyên cường độ sau scaling; chỉ window khi được yêu cầu rõ."""
    if config.intensity_encoding == "native":
        return image_xyz

    minimum = float(image_xyz.min())
    maximum = float(image_xyz.max())

    if config.intensity_encoding == "auto_hu" and (
        minimum >= -300 or maximum <= -300 or minimum < -32768 or maximum > 65535
    ):
        raise ValueError(
            f"{case_id}: cannot infer HU from scaled range [{minimum}, {maximum}]; "
            "inspect encoding and explicitly configure it"
        )

    if config.intensity_encoding == "normalized_0_1":
        if minimum < 0 or maximum > 1:
            raise ValueError(f"{case_id}: normalized_0_1 requires intensities in [0, 1]")
        return image_xyz

    low, high = config.hu_window
    return (np.clip(image_xyz, low, high) - low) / (high - low)


def to_slice_tensor(array_xyz: np.ndarray):
    """Đổi X,Y,Z → Z,Y,X; đây là đổi thứ tự lưu tensor, không reorient lần nữa."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(array_xyz.transpose(2, 1, 0)))


def resize_in_plane(
    image_zyx,
    mask_zyx,
    size_hw: tuple[int, int] | None,
):
    """Resize mỗi lát XY. Z được coi là batch, nên không bị nội suy.

    CT: bilinear, align_corners=False. Mask: nearest-exact để giữ nhãn 0/1.
    size_hw=None giữ nguyên H,W, hữu ích khi kiểm tra native resolution.
    """
    if size_hw is None or tuple(image_zyx.shape[1:]) == size_hw:
        return image_zyx, mask_zyx
    from torch.nn import functional as F

    resized_image = F.interpolate(
        image_zyx[:, None],
        size=size_hw,
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    resized_mask = None if mask_zyx is None else F.interpolate(
        mask_zyx[:, None],
        size=size_hw,
        mode="nearest-exact",
    )[:, 0]
    return resized_image, resized_mask


def read_arrays(
    case: Case,
    config: PreprocessConfig,
    *, require_mask: bool = True,
):
    """Đọc một ca thành (CT[Z,H,W], mask[Z,H,W], metadata).

    Đây là hàm điều phối preprocessing. Dataset gọi một lần khi cần chuyển
    sang volume khác; audit cũng gọi hàm này để kiểm tra CT/mask thật.
    """
    # 1. Kiểm tra headers CT/mask trước khi biến đổi bất kỳ voxel nào.
    original_ct, original_mask = inspect_pair(case, config, require_mask=require_mask)

    # 2. Thống nhất LPS: giữ ảnh LPS, đảo Z của ảnh LPI. Không nội suy.
    canonical_ct = reorient_to_lps(original_ct)
    canonical_mask = reorient_to_lps(original_mask) if original_mask is not None else None

    # 3. NiBabel áp dụng slope/intercept; kiểm tra số và nhãn ở độ phân giải gốc.
    image_xyz = canonical_ct.get_fdata(dtype=np.float32, caching="unchanged")
    mask_xyz = canonical_mask.get_fdata(dtype=np.float32, caching="unchanged") if canonical_mask is not None else None
    validate_voxels(image_xyz, mask_xyz, case.case_id)
    intensity_range = [float(image_xyz.min()), float(image_xyz.max())]
    label_values = np.unique(mask_xyz).astype(int).tolist() if mask_xyz is not None else []
    native_lesion_voxels = int(np.count_nonzero(mask_xyz)) if mask_xyz is not None else 0

    # 4. Mặc định giữ cường độ gốc sau scaling NIfTI; không suy đoán HU.
    processed_xyz = normalize_ct(image_xyz, config, case.case_id)
    del image_xyz

    # 5. Đưa về thứ tự tensor. Mặc định giữ H,W; chỉ resize khi được yêu cầu.
    image_zyx = np.ascontiguousarray(processed_xyz.transpose(2, 1, 0))
    mask_zyx = np.ascontiguousarray(mask_xyz.transpose(2, 1, 0)) if mask_xyz is not None else None
    del processed_xyz, mask_xyz
    native_size_hw = list(image_zyx.shape[1:])
    if config.size_hw is not None and tuple(image_zyx.shape[1:]) != config.size_hw:
        import torch
        image_zyx, mask_zyx = resize_in_plane(
            torch.from_numpy(image_zyx), torch.from_numpy(mask_zyx) if mask_zyx is not None else None,
            config.size_hw,
        )
        image_zyx = image_zyx.numpy()
        mask_zyx = mask_zyx.numpy() if mask_zyx is not None else None

    # 6. Giữ cả geometry native và cấu hình xử lý để truy vết/restore prediction.
    metadata = geometry_metadata(original_ct, canonical_ct, config.assume_mm)
    metadata.update(
        {
            "processed_size_hw": list(image_zyx.shape[1:]),
            "native_size_hw": native_size_hw,
            "scaled_intensity_range": intensity_range,
            "intensity_encoding": config.intensity_encoding,
            "label_values": label_values,
            "native_lesion_voxels": native_lesion_voxels,
            "nifti_scaling": {
                "slope": float(original_ct.dataobj.slope),
                "intercept": float(original_ct.dataobj.inter),
            },
            "hu_window": list(config.hu_window) if config.hu_window is not None else None,
            "padding": "replicate_boundary",
        }
    )
    return image_zyx, mask_zyx, metadata


def read_volume(case: Case, config: PreprocessConfig):
    """Torch API from Duong, backed by the same NumPy core as main's API."""
    import torch
    image, mask, metadata = read_arrays(case, config)
    return torch.from_numpy(image), torch.from_numpy(mask), metadata
