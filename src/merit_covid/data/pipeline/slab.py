"""Shared native-slice sampling along the canonical affine's axial column.

World coordinates are NiBabel RAS+ mm; array axes are LPS. Oblique grids use
slice-centre displacement along the affine column, not the world Z coordinate.
"""
import numpy as np

def slab_indices(master, depth, slab_size):
    half = slab_size // 2
    return np.clip(np.arange(master - half, master + half + 1), 0, depth - 1)

def slice_positions(indices, metadata):
    shape = metadata['canonical_shape_xyz']
    centers = np.column_stack((np.full(len(indices), (shape[0] - 1) / 2), np.full(len(indices), (shape[1] - 1) / 2), indices))
    affine = np.asarray(metadata['canonical_affine_mm'])
    return np.asarray(centers @ affine[:3, :3].T + affine[:3, 3])

def select_slices(master, volume_depth, slab_size, metadata, *, sampling_mode="index",
                  target_spacing_mm=None, boundary_policy="edge", duplicate_slice_policy="unique"):
    if type(slab_size) is not int or slab_size not in (1, 3, 5, 11):
        raise ValueError("Unsupported slab_size")
    if isinstance(master, (bool, np.bool_)) or not isinstance(master, (int, np.integer)) or not 0 <= master < volume_depth:
        raise IndexError(master)
    affine = np.asarray(metadata["canonical_affine_mm"], dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Sampling needs a finite 4x4 affine in mm")
    step = float(np.linalg.norm(affine[:3, 2]))
    if step <= 0:
        raise ValueError("Invalid axial slice-centre step")
    local_master = slab_size // 2
    offsets = np.arange(slab_size) - local_master
    if sampling_mode == "index":
        if target_spacing_mm is not None:
            raise ValueError("target_spacing_mm is only valid in physical_mm mode")
        requested_offsets = offsets * step
        native_targets = master + offsets.astype(np.float64)
    elif sampling_mode == "physical_mm":
        if boundary_policy != "edge":
            raise ValueError("physical_mm nearest sampling requires edge boundary_policy")
        if (type(target_spacing_mm) not in (int, float) or not np.isfinite(target_spacing_mm)
                or target_spacing_mm <= 0):
            raise ValueError("physical_mm requires explicit positive target_spacing_mm")
        requested_offsets = offsets * target_spacing_mm
        native_targets = master + requested_offsets / step
    else:
        raise ValueError("Unknown sampling_mode")
    in_bounds = (native_targets >= -1e-7) & (native_targets <= volume_depth - 1 + 1e-7)
    # Ties within 1e-7 voxel choose the lower canonical index.
    lower = np.floor(native_targets)
    nearest = (lower + ((native_targets - lower) > 0.5 + 1e-7)).astype(np.int64)
    if boundary_policy == "edge":
        indices = np.clip(nearest, 0, volume_depth - 1)
    elif boundary_policy == "reflect":
        if volume_depth == 1:
            indices = np.zeros_like(nearest)
        else:
            folded = nearest % (2 * (volume_depth - 1))
            indices = np.minimum(folded, 2 * (volume_depth - 1) - folded)
    else:
        raise ValueError("Unknown boundary_policy")
    indices[local_master] = master
    return indices, requested_offsets.astype(np.float32), slice_validity(
        indices, local_master, in_bounds, duplicate_slice_policy)


def slice_validity(indices, local_master, in_bounds, policy):
    if policy == "all":
        return np.ones(len(indices), dtype=bool)  # Explicit legacy loss ablation.
    if policy != "unique":
        raise ValueError("Unknown duplicate_slice_policy")
    valid = np.zeros(len(indices), dtype=bool)
    valid[local_master] = True
    seen = {int(indices[local_master])}
    # Prefer in-bounds representatives. A clamped endpoint is still a REAL
    # slice: if it occurs only at out-of-range requests, supervise it once.
    for i in sorted(range(len(indices)), key=lambda i: (not in_bounds[i], abs(i - local_master), i)):
        if int(indices[i]) not in seen:
            seen.add(int(indices[i]))
            valid[i] = True
    return valid


def sample_arrays(image_zyx, mask_zyx, master, depth, metadata, *, indices=None,
                  sampling_mode="index", target_spacing_mm=None, boundary_policy="edge",
                  duplicate_slice_policy="unique"):
    selected, requested, valid = select_slices(master, len(image_zyx), depth, metadata,
        sampling_mode=sampling_mode, target_spacing_mm=target_spacing_mm,
        boundary_policy=boundary_policy, duplicate_slice_policy=duplicate_slice_policy)
    if indices is not None:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.shape != (depth,) or np.any(selected < 0) or np.any(selected >= len(image_zyx)) or selected[depth // 2] != master:
            raise ValueError("Explicit slice indices are invalid")
        valid = slice_validity(selected, depth // 2, np.ones(depth, dtype=bool), duplicate_slice_policy)
    positions = slice_positions(selected, metadata)
    affine = np.asarray(metadata['canonical_affine_mm'])
    direction = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    result = {
        "image_slab": image_zyx[selected][None], "master_slice_index": int(master),
        "master_index": depth // 2, "slice_indices": selected.astype(np.int64),
        "slice_positions_mm": positions,
        "z_offsets_mm": ((positions - positions[depth // 2]) @ direction).astype(np.float32),
        "requested_offsets_mm": requested, "slice_valid": valid,
        "valid_region": np.broadcast_to(valid[None, :, None, None], (1, depth, *image_zyx.shape[1:])).copy(),
        "sampling_mode": sampling_mode, "target_spacing_mm": target_spacing_mm,
    }
    if mask_zyx is not None:
        result["mask_slab"] = mask_zyx[selected][None].copy().astype(np.float32)
        result["master_mask"] = result["mask_slab"][:, depth // 2].copy()
    return result
