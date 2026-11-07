"""Slice Dataset with lazy, bounded volume caching and an optional torch bridge."""

from collections import OrderedDict
import random

import nibabel as nib
import numpy as np

from merit_covid.config import Config
from .manifest import Case, validate_cases
from .nifti_io import load_case
from .preprocessing import preprocess
from .slab import sample_slab


class SlabDataset:
    """Map-style Dataset accepted by torch DataLoader without a torch import.

    Sample arrays are NumPy until the DataLoader's default collation converts them.
    Metadata stays on the dataset so heterogeneous geometry never breaks collation.
    """

    def __init__(self, cases: list[Case], config: Config, *, mode: str,
                 augmentation=None, cache_size: int = 1, assumed_unit: str | None = None):
        if mode == "inference":
            from dataclasses import replace
            cases = [replace(case, infection_mask=None, lung_mask=None) for case in cases]
        validate_cases(cases)
        if mode not in ("train", "validation", "internal_test", "external_test", "inference"):
            raise ValueError("Unknown dataset mode")
        if mode != "train" and augmentation is not None:
            raise ValueError("Random augmentation is allowed only in train mode")
        if any(c.dataset == "zenodo20" for c in cases) and mode not in ("external_test", "inference"):
            raise ValueError("Zenodo20 is external only")
        if mode == "external_test" and any(c.dataset != "zenodo20" for c in cases):
            raise ValueError("external_test requires Zenodo20 only")
        if mode != "inference" and any(c.infection_mask is None for c in cases):
            raise ValueError("Labeled mode requires infection masks")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.cases, self.config, self.mode = cases, config, mode
        self.augmentation, self.cache_size, self.assumed_unit = augmentation, cache_size, assumed_unit
        self._cache = OrderedDict()
        self.index = []
        for i, case in enumerate(cases):
            image = nib.load(str(case.image))
            if len(image.shape) != 3:
                raise ValueError(f"{case.case_id}: expected 3D CT")
            axes = nib.orientations.io_orientation(image.affine)
            superior_axis = int(np.flatnonzero(axes[:, 0] == 2)[0])
            self.index.extend((i, z) for z in range(image.shape[superior_axis]))

    def __len__(self):
        return len(self.index)

    def prepared_volume(self, case_index: int):
        if case_index not in self._cache:
            case = self.cases[case_index]
            volume = load_case(case, require_mask=self.mode != "inference", assumed_unit=self.assumed_unit)
            self._cache[case_index] = preprocess(volume, self.config, intensity=case.intensity)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(case_index)
        return self._cache[case_index]

    def __getitem__(self, index: int):
        case_index, z = self.index[index]
        prepared = self.prepared_volume(case_index)
        sample = sample_slab(prepared.image, prepared.mask, z, prepared.affine,
                             mm_per_unit=prepared.volume.mm_per_unit,
                             num_slices=self.config.num_slices, master_index=self.config.master_index,
                             boundary=self.config.boundary)
        case = self.cases[case_index]
        sample.update(case_id=case.case_id, dataset=case.dataset)
        if self.augmentation is not None:
            sample = self.augmentation(sample)
        return sample


def _seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_dataloader(dataset: SlabDataset, *, batch_size: int | None = None,
                    num_workers: int | None = None):
    try:
        import torch
    except ImportError as exc:
        raise ImportError('DataLoader requires optional training dependencies: pip install -e ".[training]"') from exc
    generator = torch.Generator().manual_seed(dataset.config.seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size or dataset.config.batch_size,
        shuffle=dataset.mode == "train",
        num_workers=dataset.config.num_workers if num_workers is None else num_workers,
        generator=generator, worker_init_fn=_seed_worker,
    )
