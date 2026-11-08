"""Shared preprocessing settings; start here when changing the input policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """Input policy shared by training, validation and testing.

    size_hw=None preserves each slice's native size without interpolation.
    Set (height, width) only for explicit in-plane resizing; Z is excluded.
    Native intensity preserves NIfTI-scaled values without windowing/normalizing.
    assume_mm permits a recorded assumption for unknown header units; it does
    not verify units or modify original files.
    """

    slab_size: int = 11
    size_hw: tuple[int, int] | None = None
    hu_window: tuple[float, float] | None = None
    intensity_encoding: str = "native"
    assume_mm: bool = False
    sampling_mode: str = "index"
    target_spacing_mm: float | None = None
    boundary_policy: str = "edge"
    duplicate_slice_policy: str = "unique"
    intensity_provenance: str | None = None

    def __post_init__(self) -> None:
        if type(self.slab_size) is not int or self.slab_size not in (1, 3, 5, 11):
            raise ValueError("slab_size must be 1, 3, 5 or 11")
        if self.sampling_mode not in ("index", "physical_mm"):
            raise ValueError("sampling_mode must be index or physical_mm")
        if self.sampling_mode == "physical_mm":
            if (type(self.target_spacing_mm) not in (int, float)
                    or not np.isfinite(self.target_spacing_mm) or self.target_spacing_mm <= 0):
                raise ValueError("physical_mm requires explicit positive target_spacing_mm")
        elif self.target_spacing_mm is not None:
            raise ValueError("target_spacing_mm is only valid in physical_mm mode")
        if self.boundary_policy not in ("edge", "reflect"):
            raise ValueError("boundary_policy must be edge or reflect")
        if self.sampling_mode == "physical_mm" and self.boundary_policy != "edge":
            raise ValueError("physical_mm nearest sampling requires edge boundary_policy")
        if self.duplicate_slice_policy not in ("unique", "all"):
            raise ValueError("duplicate_slice_policy must be unique or all")
        if type(self.assume_mm) is not bool:
            raise ValueError("assume_mm must be boolean")
        if self.intensity_provenance is not None and (
                not isinstance(self.intensity_provenance, str) or not self.intensity_provenance.strip()):
            raise ValueError("intensity_provenance must be a nonempty source description")

        if self.size_hw is not None:
            if len(self.size_hw) != 2 or any(
                type(value) is not int or value <= 0 for value in self.size_hw
            ):
                raise ValueError("size_hw must be two positive integers, or None")
            object.__setattr__(self, "size_hw", tuple(self.size_hw))

        if self.intensity_encoding not in ("native", "auto_hu", "hu", "normalized_0_1"):
            raise ValueError("Unknown intensity encoding")

        if self.hu_window is not None and (
            len(self.hu_window) != 2
            or not np.isfinite(self.hu_window).all()
            or self.hu_window[0] >= self.hu_window[1]
        ):
            raise ValueError("Invalid HU window")

        if self.intensity_encoding in ("auto_hu", "hu"):
            if self.hu_window is None:
                raise ValueError("HU processing requires an explicit hu_window")
        elif self.hu_window is not None:
            raise ValueError("hu_window requires intensity_encoding='hu' or 'auto_hu'")
