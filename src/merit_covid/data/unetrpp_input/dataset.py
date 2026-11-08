"""Lazy patches backed by the existing LPS/native reader, with optional true 3D GT."""

from bisect import bisect_right
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from merit_covid.data.manifest import load_manifest
from merit_covid.data.pipeline.cases import Case, identifier
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.geometry import inspect_pair, canonical_shape
from merit_covid.data.pipeline.preprocessing import read_arrays
from merit_covid.data.pipeline.splits import validate_splits

from .config import PatchConfig
from .tiling import PatchPlan, integer_index, extract_patch, valid_region_mask


def load_patch_cases(manifest, *, patient_map=None):
    """Reuse the manifest parser without requiring/opening any GT file.

    Retain recorded mask names for the EXISTING split fingerprint. The Dataset
    strips mask paths before every header/voxel read in inference mode.
    """
    records = load_manifest(manifest, dataset="covid19_20", check_files=False)
    if any(c.dataset != "covid19_20" for c in records):
        raise ValueError("This adapter accepts only primary covid19_20 cases")
    mapping = None
    if patient_map is not None:
        mapping = json.loads(Path(patient_map).read_text(encoding="utf-8-sig"))
        if not isinstance(mapping, dict) or set(mapping) != {c.case_id for c in records}:
            raise ValueError("Patient map must cover exactly every manifest case")
    cases = [Case(c.case_id, c.image, c.infection_mask,
                  identifier(mapping[c.case_id] if mapping is not None else
                             c.patient_id or c.group_id or c.case_id, "group_id")) for c in records]
    return sorted(cases, key=lambda c: c.case_id)


class UNETRPPPatchDataset:
    """Map-style NumPy Dataset; constructor reads headers, cache holds ONE volume.

    require_mask=False is strict CT-only inference, even if manifest contains GT.
    No patch is selected/dropped according to lesions. Valid mask is geometry only.
    """

    def __init__(self, cases, config=None, *, require_mask=False):
        if type(require_mask) is not bool:
            raise ValueError("require_mask must be boolean")
        self.config = config or PatchConfig()
        self.require_mask = require_mask
        self.cases = list(cases)
        if not self.cases or len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("Dataset needs nonempty, unique case IDs")
        if not require_mask:
            self.cases = [replace(c, mask=None) for c in self.cases]
        self.preprocess = PreprocessConfig(assume_mm=self.config.assume_mm)
        self.plans, self.ends = [], [0]
        for case in self.cases:
            ct, _ = inspect_pair(case, self.preprocess, require_mask=require_mask)
            plan = PatchPlan(tuple(reversed(canonical_shape(ct))), self.config)
            self.plans.append(plan)
            self.ends.append(self.ends[-1] + len(plan))
        self.partition = None
        self.selection = {"source": "explicit case list"}
        self._cached_index = None
        self._cached = None

    @classmethod
    def from_split(cls, manifest, split_path, fold=0, partition="train", *,
                   config=None, require_mask=False, case_ids=None, patient_map=None):
        if type(fold) is not int or fold not in range(5) or partition not in ("train", "validation", "test"):
            raise ValueError("Choose fold 0..4 and train/validation/test")
        cases = load_patch_cases(manifest, patient_map=patient_map)
        if any(c.mask is None for c in cases):
            raise ValueError("Split fingerprint needs recorded mask filenames; inference never opens those files")
        path = Path(split_path)
        raw = path.read_bytes()
        split = json.loads(raw.decode("utf-8-sig"))
        validate_splits(split, cases)
        part = next(f for f in split["folds"] if f["fold"] == fold)[partition]
        selected = set(part["case_ids"])
        if case_ids is not None:
            if (not isinstance(case_ids, (list, tuple)) or not case_ids
                    or any(not isinstance(c, str) for c in case_ids)
                    or len(set(case_ids)) != len(case_ids) or not set(case_ids) <= selected):
                raise ValueError("case_ids must be a nonempty unique subset of the chosen partition")
            selected = set(case_ids)
        dataset = cls([c for c in cases if c.case_id in selected], config, require_mask=require_mask)
        dataset.partition = partition
        dataset.selection = {
            "source": "existing canonical split", "fold": fold, "partition": partition,
            "split_manifest_fingerprint": split["manifest_fingerprint"],
            "split_sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
            "selected_case_ids": sorted(selected),
            "identity": deepcopy(split.get("identity", {})),
        }
        return dataset

    def __len__(self):
        return self.ends[-1]

    def __getstate__(self):
        return {**self.__dict__, "_cached_index": None, "_cached": None}

    def case_patch_range(self, case_index):
        case_index = integer_index(case_index, len(self.cases))
        return range(self.ends[case_index], self.ends[case_index + 1])

    def _load_case(self, case_index):
        if self._cached_index != case_index:
            self._cached = None  # Release the previous CT BEFORE decoding the next.
            self._cached_index = None
            self._cached = read_arrays(self.cases[case_index], self.preprocess,
                                       require_mask=self.require_mask)
            if self._cached[0].shape != self.plans[case_index].shape_zyx:
                self._cached = None
                raise ValueError("Volume shape changed after indexing")
            self._cached_index = case_index
        return self._cached

    def __getitem__(self, index):
        index = integer_index(index, len(self))
        case_index = bisect_right(self.ends, index) - 1
        local_index = index - self.ends[case_index]
        plan, case = self.plans[case_index], self.cases[case_index]
        location = plan[local_index]
        image, mask, geometry = self._load_case(case_index)
        metadata = {
            **deepcopy(geometry), "case_id": case.case_id, "group_id": case.group_id,
            "volume_shape_zyx": list(plan.shape_zyx), "patch_count": len(plan),
            "patch": location.to_dict(), "input_policy": self.config.to_dict(),
            "selection": deepcopy(self.selection),
            "ground_truth_read": self.require_mask,
        }
        # The shared reader's legacy slab padding field is not a patch policy.
        metadata.pop("padding", None)
        if not self.require_mask:
            metadata.pop("label_values", None)
            metadata.pop("native_lesion_voxels", None)
        sample = {
            "image_patch": extract_patch(image, location, self.config)[None],
            "valid_region": valid_region_mask(location)[None],
            "case_id": case.case_id, "group_id": case.group_id,
            "patch_index": local_index, "metadata": metadata,
        }
        if self.require_mask:
            sample["mask_patch"] = extract_patch(mask, location, self.config, is_mask=True)[None].astype(np.float32)
        return sample


def make_patch_dataloader(dataset, *, batch_size=1, num_workers=0, shuffle=False, seed=42):
    """Reuse MERGE collation and worker seeding; never pad/resize in collation."""
    import torch
    from merit_covid.data.pipeline.dataloader import collate_slabs, seed_worker
    for name, value, minimum in (("batch_size", batch_size, 1), ("num_workers", num_workers, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {name}")
    if shuffle and getattr(dataset, "partition", None) in ("validation", "test"):
        raise ValueError("Validation/test patch loaders preserve deterministic order")
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
        collate_fn=collate_slabs, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed), pin_memory=False,
    )
