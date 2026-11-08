"""Deterministic full-volume coverage and crops; no image interpolation."""

from dataclasses import dataclass
from math import prod
from numbers import Integral

import numpy as np

from .config import PatchConfig, positive_triplet


def integer_index(value, length):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise IndexError("Index must be an integer")
    value = int(value)
    if not 0 <= value < length:
        raise IndexError(value)
    return value


def axis_starts(length, size, stride):
    """Right-align the last window; short axes need just one padded crop."""
    if any(type(v) is not int or v < 1 for v in (length, size, stride)) or stride > size:
        raise ValueError("Axis length/size/stride must be positive and stride <= size")
    if length <= size:
        return (0,)
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return tuple(starts)


@dataclass(frozen=True)
class PatchLocation:
    index: int
    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]
    patch_size: tuple[int, int, int]

    @property
    def valid_shape(self):
        return tuple(b - a for a, b in zip(self.start_zyx, self.stop_zyx))

    @property
    def source_slices(self):
        return tuple(slice(a, b) for a, b in zip(self.start_zyx, self.stop_zyx))

    @property
    def valid_slices(self):
        return tuple(slice(0, n) for n in self.valid_shape)

    @property
    def padding(self):
        return tuple((0, p - n) for p, n in zip(self.patch_size, self.valid_shape))

    def to_dict(self):
        return {
            "index": self.index, "start_zyx": list(self.start_zyx),
            "stop_zyx": list(self.stop_zyx), "patch_size": list(self.patch_size),
            "valid_shape_zyx": list(self.valid_shape),
            "padding_zyx": [list(pair) for pair in self.padding],
        }


class PatchPlan:
    """Stores axis indices only, not voxels or one object per patch."""

    def __init__(self, shape_zyx, config=None):
        self.shape_zyx = positive_triplet(shape_zyx, "shape_zyx")
        self.config = config or PatchConfig()
        self.starts = tuple(axis_starts(n, p, s) for n, p, s in zip(
            self.shape_zyx, self.config.patch_size, self.config.effective_stride))

    def __len__(self):
        return prod(len(axis) for axis in self.starts)

    def __getitem__(self, index):
        index = integer_index(index, len(self))
        z, rem = divmod(index, len(self.starts[1]) * len(self.starts[2]))
        y, x = divmod(rem, len(self.starts[2]))
        start = tuple(axis[i] for axis, i in zip(self.starts, (z, y, x)))
        stop = tuple(min(a + p, n) for a, p, n in zip(start, self.config.patch_size, self.shape_zyx))
        return PatchLocation(index, start, stop, self.config.patch_size)


def valid_region_mask(location):
    valid = np.zeros(location.patch_size, dtype=bool)
    valid[location.valid_slices] = True
    return valid


def extract_patch(array_zyx, location, config, *, is_mask=False):
    array = np.asarray(array_zyx)
    if array.ndim != 3 or location.patch_size != config.patch_size:
        raise ValueError("Expected a 3D volume and matching patch configuration")
    if any(not 0 <= a < b <= n or b - a > p for a, b, n, p in zip(
            location.start_zyx, location.stop_zyx, array.shape, location.patch_size)):
        raise ValueError("Patch coordinates outside source volume")
    crop = array[location.source_slices]
    if any(after for _, after in location.padding):
        mode = "constant" if is_mask else config.padding_mode
        kwargs = {"constant_values": 0 if is_mask else config.padding_value} if mode == "constant" else {}
        crop = np.pad(crop, location.padding, mode=mode, **kwargs)
    # A consumer modifying its sample cannot modify the cached source volume.
    return np.array(crop, copy=True, order="C")
