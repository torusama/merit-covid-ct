"""Spatially ordered context sampling along the canonical superior axis."""

import numpy as np
import nibabel as nib


def sample_slab(image: np.ndarray, mask: np.ndarray | None, slice_index: int,
                affine: np.ndarray, *, mm_per_unit: float = 1.0,
                num_slices: int = 11, master_index: int = 5,
                boundary: str = "edge") -> dict:
    if image.ndim != 3 or min(image.shape) < 1 or not isinstance(slice_index, (int, np.integer)) or not 0 <= slice_index < image.shape[2]:
        raise ValueError("Expected a non-empty volume and valid master slice index")
    if num_slices not in (1, 3, 5, 11) or master_index != num_slices // 2:
        raise ValueError("Use a supported odd slab depth and centered master_index")
    if mask is not None and mask.shape != image.shape:
        raise ValueError("Image/mask shape mismatch")
    affine = np.asarray(affine)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.isfinite(mm_per_unit) or mm_per_unit <= 0:
        raise ValueError("Valid affine and physical unit scale required")
    # RAS axis direction is established by the paired NIfTI loader.
    step_mm = float(np.linalg.norm(affine[:3, 2]) * mm_per_unit)
    if step_mm <= 0:
        raise ValueError("Slice step must be positive")
    if tuple(nib.aff2axcodes(affine)) != ("R", "A", "S"):
        raise ValueError("Slab sampling requires canonical RAS geometry from load_case")
    requested = np.arange(num_slices) + slice_index - master_index
    depth = image.shape[2]
    if boundary == "edge":
        indices = np.clip(requested, 0, depth - 1)
    elif boundary == "reflect":
        if depth == 1:
            indices = np.zeros_like(requested)
        else:
            period = 2 * (depth - 1)
            folded = requested % period
            indices = np.minimum(folded, period - folded)
    else:
        raise ValueError("boundary must be edge or reflect")
    result = {
        "image": np.moveaxis(image[:, :, indices], 2, 0)[None].astype(np.float32),
        "master_index": master_index,
        "slice_index": int(slice_index),
        "z_offsets_mm": ((indices - slice_index) * step_mm).astype(np.float32),
        "context_indices": indices.astype(np.int64),
    }
    if mask is not None:
        result["mask"] = mask[:, :, slice_index][None].astype(np.float32)
    return result
