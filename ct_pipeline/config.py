"""Cấu hình đầu vào: đọc file này trước khi thay đổi preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """Các lựa chọn của pipeline, dùng chung cho train/validation/test.

    size_hw=None mặc định giữ nguyên kích thước MỖI LÁT, không nội suy.
    Chỉ truyền (height, width) khi chủ động yêu cầu resize; không gồm chiều Z.
    assume_mm chỉ cho phép xử lý header thiếu đơn vị bằng giả định mm;
    nó không xác minh đơn vị hoặc sửa file gốc.
    """

    slab_size: int = 11
    size_hw: tuple[int, int] | None = None
    hu_window: tuple[float, float] = (-1000.0, 400.0)
    intensity_encoding: str = "auto_hu"
    assume_mm: bool = False

    def __post_init__(self) -> None:
        if self.slab_size not in (1, 3, 5, 11):
            raise ValueError("slab_size must be 1, 3, 5 or 11")

        if self.size_hw is not None:
            if len(self.size_hw) != 2 or any(
                type(value) is not int or value <= 0 for value in self.size_hw
            ):
                raise ValueError("size_hw must be two positive integers, or None")

        if (
            len(self.hu_window) != 2
            or not np.isfinite(self.hu_window).all()
            or self.hu_window[0] >= self.hu_window[1]
        ):
            raise ValueError("Invalid HU window")

        if self.intensity_encoding not in ("auto_hu", "hu", "normalized_0_1"):
            raise ValueError("Unknown intensity encoding")
