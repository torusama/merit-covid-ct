"""Main API backed by the shared Duong intensity, resize and inverse functions."""
from dataclasses import dataclass
import warnings
import numpy as np
from merit_covid.config import Config
from .nifti_io import Volume
from .pipeline.config import PreprocessConfig
from .pipeline.preprocessing import normalize_ct, resize_in_plane
from .pipeline.geometry import restore_volume

def normalize(image, policy, *, intensity, hu_window=None):
    if not np.isfinite(image).all():
        raise ValueError('CT intensities must be finite')
    modes = {'none': 'native', 'hu_window': 'hu', 'normalized_0_1': 'normalized_0_1', 'auto_hu': 'auto_hu'}
    if policy not in modes:
        raise ValueError('Unknown normalization policy')
    if policy == 'hu_window' and intensity != 'HU':
        raise ValueError("HU windowing requires verified intensity='HU'")
    config = PreprocessConfig(intensity_encoding=modes[policy], hu_window=hu_window,
                              intensity_provenance="caller asserted intensity=HU" if intensity == "HU" else None)
    return normalize_ct(np.asarray(image, dtype=np.float32), config, 'configured CT')

def resize_xy(array, size, *, order):
    """Legacy XY-size helper; Config.image_size uses H,W (Y,X)."""
    if array.ndim != 3 or len(size) != 2 or min(size) < 1 or (order not in (0, 1)):
        raise ValueError('Expected a 3D array, positive XY size, and order 0 or 1')
    if tuple(array.shape[:2]) == tuple(size):
        return array
    import torch
    value = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 1, 0))).float()
    if order == 1:
        result, _ = resize_in_plane(value, None, (size[1], size[0]))
    else:
        _, result = resize_in_plane(value, value, (size[1], size[0]))
    return result.numpy().transpose(2, 1, 0).copy()

@dataclass
class PreparedVolume:
    image: np.ndarray
    mask: np.ndarray | None
    affine: np.ndarray
    canonical_shape: tuple[int, int, int]
    volume: Volume
    lost_positive_slices: tuple[int, ...]

def preprocess(volume: Volume, config: Config, *, intensity='unknown'):
    image = normalize(volume.image, config.normalization, intensity=intensity, hu_window=config.hu_window)
    original = tuple(image.shape)
    size_xy = (config.image_size[1], config.image_size[0]) if config.image_size else original[:2]
    image = resize_xy(image, size_xy, order=1).astype(np.float32, copy=False)
    mask, lost = (volume.mask, ())
    if mask is not None:
        resized = resize_xy(mask, size_xy, order=0).astype(np.uint8, copy=False)
        lost = tuple((int(i) for i in np.flatnonzero(mask.any(axis=(0, 1)) & ~resized.any(axis=(0, 1)))))
        if lost:
            warnings.warn(f'XY resize removed all lesion pixels in slices {lost}', UserWarning)
        mask = resized
    transform = np.eye(4)
    for axis in (0, 1):
        transform[axis, axis] = original[axis] / size_xy[axis]
        transform[axis, 3] = (transform[axis, axis] - 1) / 2
    return PreparedVolume(image, mask, volume.affine @ transform, original, volume, lost)

def restore_prediction(prediction, prepared, *, binary=True):
    if prediction.shape != prepared.image.shape:
        raise ValueError('Prediction must match prepared volume shape')
    metadata = dict(prepared.volume.metadata)
    metadata['processed_size_hw'] = [prepared.image.shape[1], prepared.image.shape[0]]
    return restore_volume(np.asarray(prediction).transpose(2, 1, 0), metadata, probabilities=not binary).astype(np.uint8 if binary else np.float32)
