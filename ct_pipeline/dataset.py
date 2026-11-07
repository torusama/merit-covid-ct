"""Một mẫu Dataset = một master slice và các lát CT lân cận.

preprocessing.py xử lý cả volume; Dataset chỉ lập chỉ số master, cache một ca
và cắt slab. dataloader.py mới ghép các mẫu này thành batch.
"""

from __future__ import annotations

from bisect import bisect_right
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

# Các tên từng nằm ở dataset.py vẫn import được, để CLI/notebook cũ tiếp tục chạy.
# Code mới dùng import từ ct_pipeline hoặc từ module chuyên trách tương ứng.
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


def _slab_indices(master: int, depth: int, slab_size: int) -> np.ndarray:
    """Lấy D chỉ số quanh master; clip chỉ số để lặp lát biên."""
    half = slab_size // 2
    return np.clip(np.arange(master - half, master + half + 1), 0, depth - 1)


def _slice_positions(indices: np.ndarray, metadata: dict) -> np.ndarray:
    """Tâm các lát trên native grid, tọa độ RAS world mm."""
    shape_xyz = metadata["canonical_shape_xyz"]
    voxel_centers = np.column_stack(
        (
            np.full(len(indices), (shape_xyz[0] - 1) / 2),
            np.full(len(indices), (shape_xyz[1] - 1) / 2),
            indices,
        )
    )
    affine = np.asarray(metadata["canonical_affine_mm"])
    return np.asarray(voxel_centers @ affine[:3, :3].T + affine[:3, 3])


class CTSlabDataset(Dataset):
    """Trả một mẫu cho mọi master slice, gồm cả lát có GT mask rỗng.

    Khởi tạo chỉ đọc headers. Trong mỗi worker, cache tối đa một volume đã
    preprocessing; giải phóng cache cũ trước khi đọc volume khác.
    len(dataset) là số MASTER SLICES, không phải số CT volumes.
    """

    def __init__(
        self,
        cases: list[Case],
        config: PreprocessConfig | None = None,
    ):
        self.cases = list(cases)
        self.config = config or PreprocessConfig()
        self.partition: str | None = None

        if not self.cases or len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Dataset needs nonempty, unique cases")

        # ends chứa mốc cộng dồn số lát để đổi global index → case và master.
        self.ends = [0]
        for case in self.cases:
            ct, _ = inspect_pair(case, self.config)
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
    ) -> CTSlabDataset:
        """Chọn một partition của split đã lưu, sau khi kiểm tra leakage."""
        cases, _ = load_cases(manifest, patient_map=patient_map)
        with open(split_path, encoding="utf-8") as handle:
            split = json.load(handle)
        validate_splits(split, cases)

        if partition not in ("train", "validation", "test") or fold not in range(5):
            raise ValueError("Choose fold 0..4 and train/validation/test")

        selected_fold = next(item for item in split["folds"] if item["fold"] == fold)
        selected_ids = set(selected_fold[partition]["case_ids"])
        selected_cases = [case for case in cases if case.case_id in selected_ids]
        dataset = cls(selected_cases, config)
        dataset.partition = partition
        return dataset

    def __len__(self) -> int:
        return self.ends[-1]

    def __getstate__(self) -> dict:
        """Windows worker nhận cấu hình/index, không nhận bản copy cache CT."""
        return {
            **self.__dict__,
            "_cached_index": None,
            "_cached": None,
        }

    def _load_case(self, case_index: int):
        if self._cached_index != case_index:
            self._cached = None
            self._cached_index = None
            self._cached = read_volume(self.cases[case_index], self.config)
            self._cached_index = case_index
        return self._cached

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)

        # 1. Đổi chỉ số mẫu toàn Dataset thành một case và một master trong case.
        case_index = bisect_right(self.ends, index) - 1
        master = index - self.ends[case_index]
        case = self.cases[case_index]

        # 2. Xử lý cả volume một lần; các master tiếp theo tái sử dụng cache này.
        image_zyx, mask_zyx, metadata = self._load_case(case_index)

        # 3. Lấy context quanh master, lặp lát biên nếu cần.
        indices = _slab_indices(master, len(image_zyx), self.config.slab_size)
        positions_mm = _slice_positions(indices, metadata)
        half = self.config.slab_size // 2

        # Khoảng cách tâm có dấu dọc vector slice thực trong affine, kể cả oblique.
        affine = np.asarray(metadata["canonical_affine_mm"])
        slice_direction = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
        offsets_mm = (positions_mm - positions_mm[half]) @ slice_direction

        # 4. Hợp đồng đầu ra: một mẫu, chưa có chiều batch B.
        return {
            "image_slab": image_zyx[indices].unsqueeze(0),  # [1,D,H,W]
            "master_mask": mask_zyx[master].unsqueeze(0).clone(),  # [1,H,W]
            "master_slice_index": int(master),
            "case_id": case.case_id,
            "group_id": case.group_id,
            "z_offsets_mm": torch.tensor(offsets_mm, dtype=torch.float32),
            "slice_indices": torch.tensor(indices, dtype=torch.int64),
            "slice_positions_mm": torch.tensor(positions_mm, dtype=torch.float64),
            "metadata": metadata,
        }
