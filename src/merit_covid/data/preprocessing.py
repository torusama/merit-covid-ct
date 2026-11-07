"""Deterministic paired preprocessing with recorded inverse geometry."""

from dataclasses import dataclass
import warnings

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from merit_covid.config import Config
from .nifti_io import Volume, binary_mask


def resize_xy(array: np.ndarray, size: tuple[int, int], *, order: int) -> np.ndarray:
    """Pixel-center sampling with edge extension; axis 2 is never interpolated."""
    if array.ndim != 3 or len(size) != 2 or min(size) < 1 or order not in (0, 1):
        raise ValueError("Expected a 3D array, positive XY size, and order 0 or 1")
    if array.shape[:2] == tuple(size):
        return array.copy()
    x = (np.arange(size[0]) + 0.5) * array.shape[0] / size[0] - 0.5
    y = (np.arange(size[1]) + 0.5) * array.shape[1] / size[1] - 0.5
    coords = np.meshgrid(x, y, indexing="ij")
    return np.stack([map_coordinates(array[:, :, z], coords, order=order, mode="nearest", prefilter=False)
                     for z in range(array.shape[2])], axis=2)


def normalize(image: np.ndarray, policy: str, *, intensity: str, hu_window=(-1000.0, 400.0)) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError("CT intensities must be finite")
    if policy == "none":
        return image.copy()
    if policy == "zscore":
        return ((image - image.mean()) / max(float(image.std()), 1e-8)).astype(np.float32)
    if policy == "hu_window":
        if intensity != "HU":
            raise ValueError("HU windowing requires verified intensity='HU'")
        low, high = hu_window
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("Invalid HU window")
        return np.clip((image - low) / (high - low), 0, 1)
    if policy == "windowed_255":
        if intensity != "windowed_255" or image.min() < 0 or image.max() > 255:
            raise ValueError("windowed_255 requires verified provenance and values in [0,255]")
        return image / 255.0
    raise ValueError(f"Unknown normalization: {policy}")


@dataclass
class PreparedVolume:
    image: np.ndarray
    mask: np.ndarray | None
    affine: np.ndarray
    canonical_shape: tuple[int, int, int]
    volume: Volume
    lost_positive_slices: tuple[int, ...]


def preprocess(volume: Volume, config: Config, *, intensity: str = "unknown") -> PreparedVolume:
    image = normalize(volume.image, config.normalization, intensity=intensity, hu_window=config.hu_window)
    mask = volume.mask
    original = tuple(image.shape)
    size = config.image_size or original[:2]
    image = resize_xy(image, size, order=1).astype(np.float32)
    lost = ()
    if mask is not None:
        resized = resize_xy(mask, size, order=0).astype(np.uint8)
        lost = tuple(int(i) for i in np.flatnonzero(mask.any(axis=(0, 1)) & ~resized.any(axis=(0, 1))))
        if lost:
            warnings.warn(f"XY resize removed all lesion pixels in slices {lost}; review internal preprocessing", UserWarning)
        mask = resized
    transform = np.eye(4)
    for axis in (0, 1):
        transform[axis, axis] = original[axis] / size[axis]
        transform[axis, 3] = (transform[axis, axis] - 1) / 2
    return PreparedVolume(image, mask, volume.affine @ transform, original, volume, lost)


def restore_prediction(prediction: np.ndarray, prepared: PreparedVolume, *, binary: bool = True) -> np.ndarray:
    """Restore XY size and reverse axis permutations/flips to the original grid.

    Resizing is lossy; this restores spatial geometry, not deleted lesion detail.
    Binary inputs use nearest neighbor; probabilities use linear interpolation.
    """
    if prediction.shape != prepared.image.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction must be finite and match prepared volume shape")
    if binary:
        prediction = binary_mask(prediction)
    elif np.any((prediction < 0) | (prediction > 1)):
        raise ValueError("Probabilities must be in [0,1]")
    canonical = resize_xy(prediction.astype(np.float32), prepared.canonical_shape[:2], order=0 if binary else 1)
    inverse = nib.orientations.ornt_transform(
        nib.orientations.axcodes2ornt(("R", "A", "S")),
        nib.orientations.io_orientation(prepared.volume.original_affine),
    )
    restored = nib.orientations.apply_orientation(canonical, inverse).copy()
    return restored.astype(np.uint8 if binary else np.float32)
