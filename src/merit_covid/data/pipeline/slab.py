"""Shared NumPy sampler: exact Duong indices, Y/X layout and world offsets."""
import numpy as np

def slab_indices(master, depth, slab_size):
    half = slab_size // 2
    return np.clip(np.arange(master - half, master + half + 1), 0, depth - 1)

def slice_positions(indices, metadata):
    shape = metadata['canonical_shape_xyz']
    centers = np.column_stack((np.full(len(indices), (shape[0] - 1) / 2), np.full(len(indices), (shape[1] - 1) / 2), indices))
    affine = np.asarray(metadata['canonical_affine_mm'])
    return np.asarray(centers @ affine[:3, :3].T + affine[:3, 3])

def sample_arrays(image_zyx, mask_zyx, master, depth, metadata, *, indices=None):
    if not isinstance(master, (int, np.integer)) or not 0 <= master < len(image_zyx):
        raise IndexError(master)
    indices = slab_indices(master, len(image_zyx), depth) if indices is None else indices
    positions = slice_positions(indices, metadata)
    affine = np.asarray(metadata['canonical_affine_mm'])
    direction = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    result = {'image_slab': image_zyx[indices][None], 'master_slice_index': int(master), 'slice_indices': indices.astype(np.int64), 'slice_positions_mm': positions, 'z_offsets_mm': ((positions - positions[depth // 2]) @ direction).astype(np.float32)}
    if mask_zyx is not None:
        result['master_mask'] = mask_zyx[master][None].copy().astype(np.float32)
    return result
