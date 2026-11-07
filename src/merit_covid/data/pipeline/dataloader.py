"""Dataset trả một mẫu; file này ghép các mẫu thành batch và chọn một fold.

Điểm bắt đầu cho code train sau này: build_fold_loaders().
File không chứa model, loss, optimizer hoặc training loop.
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
    """Ghép tensor thành batch; giữ metadata từng ca trong một list riêng."""
    tensor_samples = [
        {key: value for key, value in sample.items() if key != "metadata"}
        for sample in samples
    ]
    batch = default_collate(tensor_samples)
    batch["metadata"] = [sample["metadata"] for sample in samples]
    return batch


def seed_worker(worker_id: int) -> None:
    """Hàm cấp module để Windows multiprocessing có thể pickle."""
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
    """Tạo loader CPU. Validation/test giữ thứ tự master để ghép volume."""
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
    """Ba loader cùng một fold. Truy cập bằng .train, .validation, .test."""

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
    """Đọc split ĐÃ LƯU và tạo ba loader; không tạo lại hay thay đổi split.

    Mỗi Dataset xác minh split trước khi chọn case IDs. Khởi tạo chỉ đọc headers;
    dữ liệu voxel của một ca được xử lý khi loader yêu cầu mẫu của ca đó.
    Cùng một PreprocessConfig áp dụng cho cả train/validation/test.
    """
    # Local import giữ hướng phụ thuộc rõ và hỗ trợ API cũ từ dataset.py.
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
