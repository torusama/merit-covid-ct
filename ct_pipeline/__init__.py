"""API đầu vào dùng chung: cấu hình → Dataset → DataLoader của một fold.

Ví dụ chạy được: examples/read_training_batch.py.
Đọc ct_pipeline/README.md để đi theo source của từng bước.
"""

from .config import PreprocessConfig
from .dataloader import FoldDataLoaders, build_fold_loaders, make_dataloader
from .dataset import CTSlabDataset

__all__ = [
    "PreprocessConfig",
    "CTSlabDataset",
    "FoldDataLoaders",
    "build_fold_loaders",
    "make_dataloader",
]
