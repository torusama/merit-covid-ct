"""NIfTI geometry checks, RAS reorientation, and axial slabs without Z resampling."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import random

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, default_collate

from .cases import load_cases
from .splits import validate_splits


@dataclass(frozen=True)
class PreprocessConfig:
    slab_size: int = 11
    size_hw: tuple[int, int] | None = (256, 256)
    hu_window: tuple[float, float] = (-1000.0, 400.0)
    intensity_encoding: str = "auto_hu"
    assume_mm: bool = False

    def __post_init__(self):
        if self.slab_size not in (1, 3, 5, 11):
            raise ValueError("slab_size must be 1, 3, 5 or 11")
        if self.size_hw is not None and (len(self.size_hw) != 2 or any(type(x) is not int or x <= 0 for x in self.size_hw)):
            raise ValueError("size_hw must be two positive integers, or None")
        if len(self.hu_window) != 2 or not np.isfinite(self.hu_window).all() or self.hu_window[0] >= self.hu_window[1]:
            raise ValueError("Invalid HU window")
        if self.intensity_encoding not in ("auto_hu", "hu", "normalized_0_1"):
            raise ValueError("Unknown intensity encoding")


def affine_mm(image, assume_mm=False):
    unit = image.header.get_xyzt_units()[0]
    factors = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in factors and not (unit == "unknown" and assume_mm):
        raise ValueError(f"Spatial unit is {unit!r}; use assume_mm only with documented source evidence")
    affine = np.asarray(image.affine, dtype=np.float64).copy()
    affine[:3] *= factors.get(unit, 1.0)
    return affine


def inspect_pair(case, config):
    ct, mask = nib.load(case.image), nib.load(case.mask)
    for name, image in (("CT", ct), ("mask", mask)):
        affine = image.affine
        if len(image.shape) != 3 or any(s <= 0 for s in image.shape):
            raise ValueError(f"{case.case_id}: {name} must be a nonempty 3D volume")
        if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12 or not np.allclose(affine[3], [0, 0, 0, 1]):
            raise ValueError(f"{case.case_id}: invalid {name} affine")
        zooms = np.asarray(image.header.get_zooms()[:3])
        if not np.isfinite(zooms).all() or np.any(zooms <= 0) or not np.allclose(zooms, nib.affines.voxel_sizes(affine), atol=1e-4, rtol=1e-4):
            raise ValueError(f"{case.case_id}: {name} spacing disagrees with affine")
    ct_mm, mask_mm = affine_mm(ct, config.assume_mm), affine_mm(mask, config.assume_mm)
    if (ct.shape != mask.shape or not np.allclose(ct_mm, mask_mm, atol=1e-4, rtol=0)
            or nib.aff2axcodes(ct.affine) != nib.aff2axcodes(mask.affine)):
        raise ValueError(f"{case.case_id}: CT/mask geometry mismatch (shape, affine, orientation or spacing)")
    return ct, mask


def read_volume(case, config):
    """Validate one complete pair. NiBabel applies scl_slope/scl_inter before HU checks."""
    ct, mask = inspect_pair(case, config)
    original_ornt = nib.orientations.io_orientation(ct.affine)
    canonical_ct, canonical_mask = nib.as_closest_canonical(ct), nib.as_closest_canonical(mask)
    image = canonical_ct.get_fdata(dtype=np.float32, caching="unchanged")
    labels = canonical_mask.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(image).all() or not np.isfinite(labels).all():
        raise ValueError(f"{case.case_id}: CT/mask contains NaN or Inf")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{case.case_id}: mask labels must be exactly 0/1; found {np.unique(labels)}")
    label_values = np.unique(labels).astype(int).tolist()
    lesion_voxels = int(np.count_nonzero(labels))
    lo, hi = float(image.min()), float(image.max())
    if config.intensity_encoding == "auto_hu" and (lo >= -300 or hi <= -300 or lo < -32768 or hi > 65535):
        raise ValueError(f"{case.case_id}: cannot infer HU from scaled range [{lo}, {hi}]; inspect encoding and explicitly configure it")
    if config.intensity_encoding == "normalized_0_1":
        if lo < 0 or hi > 1:
            raise ValueError(f"{case.case_id}: normalized_0_1 requires intensities in [0, 1]")
    else:
        low, high = config.hu_window
        image = (np.clip(image, low, high) - low) / (high - low)
    # Tensor convention is Z,Y,X; metadata shapes and NIfTI volumes are X,Y,Z.
    image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 1, 0)))
    labels = torch.from_numpy(np.ascontiguousarray(labels.transpose(2, 1, 0)))
    native_hw = tuple(image.shape[1:])
    if config.size_hw is not None and native_hw != config.size_hw:
        image = F.interpolate(image[:, None], size=config.size_hw, mode="bilinear", align_corners=False)[:, 0]
        labels = F.interpolate(labels[:, None], size=config.size_hw, mode="nearest-exact")[:, 0]
    canonical_mm = affine_mm(canonical_ct, config.assume_mm)
    metadata = {
        "original_shape_xyz": list(ct.shape), "canonical_shape_xyz": list(canonical_ct.shape),
        "original_affine": ct.affine.tolist(), "original_affine_mm": affine_mm(ct, config.assume_mm).tolist(),
        "canonical_affine_mm": canonical_mm.tolist(),
        "canonical_to_original_orientation": nib.orientations.ornt_transform(
            nib.orientations.io_orientation(canonical_ct.affine), original_ornt).tolist(),
        "original_orientation": list(nib.aff2axcodes(ct.affine)),
        "canonical_orientation": list(nib.aff2axcodes(canonical_ct.affine)),
        "native_spacing_mm": nib.affines.voxel_sizes(canonical_mm).tolist(),
        "processed_size_hw": list(image.shape[1:]), "native_size_hw": list(native_hw),
        "spatial_unit_in_header": ct.header.get_xyzt_units()[0], "assume_mm": config.assume_mm,
        "scaled_intensity_range": [lo, hi], "intensity_encoding": config.intensity_encoding,
        "label_values": label_values, "native_lesion_voxels": lesion_voxels,
        "nifti_scaling": {"slope": float(ct.dataobj.slope), "intercept": float(ct.dataobj.inter)},
        "hu_window": list(config.hu_window), "padding": "replicate_boundary",
        "z_resampled": False,
    }
    return image, labels, metadata


def restore_volume(prediction_zyx, metadata, probabilities=False):
    """Undo in-plane size and orientation; never interpolate along Z."""
    pred = torch.as_tensor(np.asarray(prediction_zyx), dtype=torch.float32)
    shape = metadata["canonical_shape_xyz"]
    expected = (shape[2], *metadata["processed_size_hw"])
    if tuple(pred.shape) != expected or not torch.isfinite(pred).all():
        raise ValueError(f"Prediction shape must be {expected}, finite; found {tuple(pred.shape)}")
    if not probabilities and not torch.all((pred == 0) | (pred == 1)):
        raise ValueError("Binary restoration requires 0/1 predictions")
    if probabilities and not torch.all((pred >= 0) & (pred <= 1)):
        raise ValueError("Probability restoration requires values in [0, 1]")
    if tuple(pred.shape[1:]) != tuple(metadata["native_size_hw"]):
        kwargs = {"mode": "bilinear", "align_corners": False} if probabilities else {"mode": "nearest-exact"}
        pred = F.interpolate(pred[:, None], size=metadata["native_size_hw"], **kwargs)[:, 0]
    xyz = pred.numpy().transpose(2, 1, 0)
    original = nib.orientations.apply_orientation(xyz, np.asarray(metadata["canonical_to_original_orientation"]))
    if tuple(original.shape) != tuple(metadata["original_shape_xyz"]):
        raise ValueError("Restored prediction shape disagrees with original volume")
    return np.ascontiguousarray(original)


class CTSlabDataset(Dataset):
    """All master slices, deterministic order, at most one decoded volume per worker.

    Initialization reads headers only. The one-volume cache is dropped on Windows
    worker serialization, so it does not replicate a parent volume into each worker.
    """
    def __init__(self, cases, config=None):
        self.cases = list(cases)
        self.config = config or PreprocessConfig()
        if not self.cases or len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("Dataset needs nonempty, unique cases")
        self.ends = [0]
        for case in self.cases:
            ct, _ = inspect_pair(case, self.config)
            transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(ct.affine),
                                                       nib.orientations.axcodes2ornt(("R", "A", "S")))
            canonical_shape = np.asarray(ct.shape)[np.argsort(transform[:, 0])]
            self.ends.append(self.ends[-1] + int(canonical_shape[2]))
        self._cached_index, self._cached = None, None

    @classmethod
    def from_split(cls, manifest, split_path, fold, partition, config=None, patient_map=None):
        cases, _ = load_cases(manifest, patient_map=patient_map)
        with open(split_path, encoding="utf-8") as handle:
            split = json.load(handle)
        validate_splits(split, cases)
        if partition not in ("train", "validation", "test") or fold not in range(5):
            raise ValueError("Choose fold 0..4 and train/validation/test")
        selected = next(f for f in split["folds"] if f["fold"] == fold)[partition]["case_ids"]
        result = cls([c for c in cases if c.case_id in set(selected)], config)
        result.partition = partition
        return result

    def __len__(self):
        return self.ends[-1]

    def __getstate__(self):
        return {**self.__dict__, "_cached_index": None, "_cached": None}

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        case_index = bisect_right(self.ends, index) - 1
        master = index - self.ends[case_index]
        case = self.cases[case_index]
        if self._cached_index != case_index:
            self._cached = None  # Release the old volume before decoding the next.
            self._cached_index = None
            self._cached = read_volume(case, self.config)
            self._cached_index = case_index
        image, labels, metadata = self._cached
        half = self.config.slab_size // 2
        indices = np.clip(np.arange(master - half, master + half + 1), 0, len(image) - 1)
        affine = np.asarray(metadata["canonical_affine_mm"])
        # Slice-center positions in RAS world millimetres. Signed Euclidean
        # distances along the actual affine slice vector (also valid for oblique CT).
        shape = metadata["canonical_shape_xyz"]
        points = np.column_stack((np.full(len(indices), (shape[0] - 1) / 2),
                                  np.full(len(indices), (shape[1] - 1) / 2), indices))
        positions = nib.affines.apply_affine(affine, points)
        direction = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
        offsets = (positions - positions[half]) @ direction
        return {
            "image_slab": image[indices].unsqueeze(0),
            "master_mask": labels[master].unsqueeze(0).clone(),
            "master_slice_index": int(master), "case_id": case.case_id, "group_id": case.group_id,
            "z_offsets_mm": torch.tensor(offsets, dtype=torch.float32),
            "slice_indices": torch.tensor(indices, dtype=torch.int64),
            "slice_positions_mm": torch.tensor(positions, dtype=torch.float64),
            "metadata": metadata,
        }


def collate_slabs(samples):
    # Original geometry can differ between patients. Keep metadata as a list.
    batch = default_collate([{k: v for k, v in sample.items() if k != "metadata"} for sample in samples])
    batch["metadata"] = [sample["metadata"] for sample in samples]
    return batch


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_dataloader(dataset, batch_size=1, shuffle=False, seed=42, num_workers=0):
    if getattr(dataset, "partition", None) in ("validation", "test") and shuffle:
        raise ValueError("Validation/test loaders must preserve deterministic slice order")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_slabs, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), pin_memory=False)
