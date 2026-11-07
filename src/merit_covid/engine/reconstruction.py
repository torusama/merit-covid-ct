"""Reassemble every canonical master slice exactly once, in voxel order."""

import numpy as np


def reconstruct_volume(predictions, shape: tuple[int, int, int]) -> np.ndarray:
    """Consume (slice_index, [H,W]) pairs; reject missing and duplicate slices."""
    if len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError("Expected a positive 3D shape")
    result = np.empty(shape, dtype=np.float32)
    seen = set()
    for index, values in predictions:
        if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)) or not 0 <= index < shape[2]:
            raise ValueError(f"Invalid slice index: {index}")
        if index in seen:
            raise ValueError(f"Duplicate slice index: {index}")
        values = np.asarray(values)
        if values.shape != shape[:2] or not np.isfinite(values).all():
            raise ValueError("Slice prediction must be finite and match target XY shape")
        result[:, :, index] = values
        seen.add(index)
    if len(seen) != shape[2]:
        raise ValueError(f"Missing slice indexes: {sorted(set(range(shape[2])) - seen)}")
    return result
