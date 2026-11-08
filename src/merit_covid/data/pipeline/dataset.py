"""One sample contains a master slice and its neighboring CT context.

Preprocessing handles the whole volume. This Dataset indexes masters, caches
one case and samples slabs; dataloader.py collates those samples into batches.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cases import Case, load_cases
from .config import PreprocessConfig
from .dataloader import collate_slabs, make_dataloader, seed_worker
from .geometry import affine_mm, canonical_shape, inspect_pair, restore_volume
from .preprocessing import read_volume
from .splits import validate_splits

# Preserve historical dataset.py exports for existing CLIs and notebooks.
# New code imports from merit_covid or the corresponding specialized module.
__all__ = [
    "CTSlabDataset",
    "PreprocessConfig",
    "read_volume",
    "restore_volume",
    "inspect_pair",
    "affine_mm",
    "make_dataloader",
    "collate_slabs",
    "seed_worker",
]


from .slab import slab_indices as _slab_indices, slice_positions as _slice_positions, sample_arrays


class CTSlabDataset(Dataset):
    """Expose every master slice, including slices with empty GT masks.

    Initialization reads headers only. Each worker caches at most one processed
    volume, releasing it before loading the next. Dataset length counts master
    slices, not CT volumes.
    """

    def __init__(
        self,
        cases: list[Case],
        config: PreprocessConfig | None = None,
        *, require_mask: bool = True,
    ):
        if type(require_mask) is not bool:
            raise ValueError("require_mask must be boolean")
        self.require_mask = require_mask
        self.cases = list(cases) if require_mask else [replace(c, mask=None) for c in cases]
        self.config = config or PreprocessConfig()
        self.partition: str | None = None

        if not self.cases or len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Dataset needs nonempty, unique cases")

        # Cumulative slice counts map a global sample index to case and master.
        self.ends = [0]
        for case in self.cases:
            ct, _ = inspect_pair(case, self.config, require_mask=require_mask)
            self.ends.append(self.ends[-1] + canonical_shape(ct)[2])

        self._cached_index = None
        self._cached = None

    @classmethod
    def from_split(
        cls,
        manifest: str | Path,
        split_path: str | Path,
        fold: int,
        partition: str,
        config: PreprocessConfig | None = None,
        patient_map: str | Path | None = None,
        *, require_mask: bool = True,
    ) -> CTSlabDataset:
        """Select one saved partition after validating split identity/leakage."""
        cases, identity = load_cases(manifest, patient_map=patient_map, check_masks=require_mask)
        with open(split_path, encoding="utf-8") as handle:
            split = json.load(handle)
        validate_splits(split, cases)

        if partition not in ("train", "validation", "test") or fold not in range(5):
            raise ValueError("Choose fold 0..4 and train/validation/test")

        selected_fold = next(item for item in split["folds"] if item["fold"] == fold)
        selected_ids = set(selected_fold[partition]["case_ids"])
        selected_cases = [case for case in cases if case.case_id in selected_ids]
        dataset = cls(selected_cases, config, require_mask=require_mask)
        dataset.partition = partition
        dataset.identity = identity
        dataset.split_fingerprint = split["manifest_fingerprint"]
        return dataset

    def __len__(self) -> int:
        return self.ends[-1]

    def __getstate__(self) -> dict:
        """Send configuration/index to Windows workers, not cached CT arrays."""
        return {
            **self.__dict__,
            "_cached_index": None,
            "_cached": None,
        }

    def _load_case(self, case_index: int):
        if self._cached_index != case_index:
            self._cached = None
            self._cached_index = None
            self._cached = (read_volume(self.cases[case_index], self.config) if self.require_mask else
                            read_volume(self.cases[case_index], self.config, require_mask=False))
            self._cached_index = case_index
        return self._cached

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)

        # 1. Map the global sample index to a case and its local volume master.
        case_index = bisect_right(self.ends, index) - 1
        master = index - self.ends[case_index]
        case = self.cases[case_index]

        # 2. Process a volume once; subsequent masters reuse its cached arrays.
        image_zyx, mask_zyx, metadata = self._load_case(case_index)

        # 3. Sample master context with the configured boundary/duplicate policy.
        sample = sample_arrays(image_zyx.numpy(), mask_zyx.numpy() if mask_zyx is not None else None,
            master, self.config.slab_size, metadata, sampling_mode=self.config.sampling_mode,
            target_spacing_mm=self.config.target_spacing_mm, boundary_policy=self.config.boundary_policy,
            duplicate_slice_policy=self.config.duplicate_slice_policy)
        metadata = {**metadata, "case_id": case.case_id, "group_id": case.group_id,
                    "sampling": {"mode": self.config.sampling_mode, "target_spacing_mm": self.config.target_spacing_mm,
                                 "boundary_policy": self.config.boundary_policy,
                                 "duplicate_slice_policy": self.config.duplicate_slice_policy}}
        return {**{key:torch.from_numpy(value) if isinstance(value,np.ndarray) else value for key,value in sample.items()},
                'case_id':case.case_id,'group_id':case.group_id,'metadata':metadata}
