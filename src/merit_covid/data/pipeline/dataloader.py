"""Collate Dataset samples and construct loaders for one saved fold.

build_fold_loaders() is the shared Tensor API entry point. Models, losses and
training loops are separate from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from .config import PreprocessConfig


def collate_slabs(samples: list[dict]) -> dict:
    """Stack tensors while retaining per-case metadata in a separate list."""
    if not samples or any(set(sample) != set(samples[0]) for sample in samples):
        raise ValueError("Nonempty samples with consistent keys required")
    nullable = {key for key in samples[0] if any(sample[key] is None for sample in samples)}
    tensor_samples = [
        {key: value for key, value in sample.items() if key != "metadata" and key not in nullable}
        for sample in samples
    ]
    batch = default_collate(tensor_samples)
    batch["metadata"] = [sample["metadata"] for sample in samples]
    batch.update({key: [sample[key] for sample in samples] for key in nullable})
    return batch


def seed_worker(worker_id: int) -> None:
    """Module-level initializer so Windows multiprocessing can pickle it."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Build a CPU loader; validation/test preserve master reconstruction order."""
    if getattr(dataset, "partition", None) in ("validation", "test") and shuffle:
        raise ValueError("Validation/test loaders must preserve deterministic slice order")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_slabs,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=False,
    )


@dataclass(frozen=True)
class FoldDataLoaders:
    """One fold's loaders, exposed as .train, .validation and .test."""

    fold: int
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def build_fold_loaders(
    manifest: str | Path,
    split_path: str | Path,
    fold: int = 0,
    *,
    preprocess: PreprocessConfig | None = None,
    batch_size: int = 2,
    shuffle_train: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    patient_map: str | Path | None = None,
) -> FoldDataLoaders:
    """Read the saved split and build three loaders without rewriting partitions.

    Each Dataset validates the split before selecting case IDs. Initialization
    reads headers only; voxel processing is lazy. All three partitions share
    the same PreprocessConfig.
    """
    # Local import avoids a cycle with historical dataset.py re-exports.
    from .dataset import CTSlabDataset

    config = preprocess or PreprocessConfig()
    loaders = {}
    for partition in ("train", "validation", "test"):
        dataset = CTSlabDataset.from_split(
            manifest=manifest,
            split_path=split_path,
            fold=fold,
            partition=partition,
            config=config,
            patient_map=patient_map,
        )
        loaders[partition] = make_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle_train if partition == "train" else False,
            seed=seed,
            num_workers=num_workers,
        )

    return FoldDataLoaders(fold=fold, **loaders)
