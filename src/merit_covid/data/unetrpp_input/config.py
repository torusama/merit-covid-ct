"""Explicit patch extraction policy; no intensity or spatial resampling policy."""

from dataclasses import dataclass
import math


def positive_triplet(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must be three positive integers in Z,Y,X order")
    if any(type(v) is not int or v < 1 for v in value):
        raise ValueError(f"{name} must be three positive integers in Z,Y,X order")
    return tuple(value)


@dataclass(frozen=True)
class PatchConfig:
    patch_size: tuple[int, int, int] = (64, 128, 128)
    stride: tuple[int, int, int] | None = None
    overlap: tuple[float, float, float] | None = None
    padding_mode: str = "edge"
    padding_value: float = 0.0
    assume_mm: bool = False

    def __post_init__(self):
        size = positive_triplet(self.patch_size, "patch_size")
        object.__setattr__(self, "patch_size", size)
        if self.stride is not None and self.overlap is not None:
            raise ValueError("Specify stride OR overlap, not both")
        if self.overlap is not None:
            if (not isinstance(self.overlap, (tuple, list)) or len(self.overlap) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1
                           for v in self.overlap)):
                raise ValueError("overlap must contain three finite fractions in [0,1)")
            object.__setattr__(self, "overlap", tuple(float(v) for v in self.overlap))
        if self.stride is not None:
            stride = positive_triplet(self.stride, "stride")
            if any(s > p for s, p in zip(stride, size)):
                raise ValueError("stride cannot exceed patch_size: that would leave coverage gaps")
            object.__setattr__(self, "stride", stride)
        if self.padding_mode not in ("edge", "constant"):
            raise ValueError("padding_mode must be edge or constant")
        if type(self.padding_value) not in (int, float) or not math.isfinite(self.padding_value):
            raise ValueError("padding_value must be finite")
        if type(self.assume_mm) is not bool:
            raise ValueError("assume_mm must be boolean")

    @property
    def effective_stride(self):
        if self.stride is not None:
            return self.stride
        if self.overlap is not None:
            return tuple(max(1, math.floor(p * (1 - o))) for p, o in zip(self.patch_size, self.overlap))
        return self.patch_size

    def to_dict(self):
        return {
            "type": "3d_crop_tiling", "axes": "Z,Y,X", "patch_size": list(self.patch_size),
            "requested_stride": list(self.stride) if self.stride is not None else None,
            "requested_overlap": list(self.overlap) if self.overlap is not None else None,
            "effective_stride": list(self.effective_stride),
            "end_alignment": "append final window ending at the real volume boundary",
            "padding_mode": self.padding_mode,
            "padding_value": self.padding_value if self.padding_mode == "constant" else None,
            "padding_placement": "after each short axis only", "mask_padding_value": 0,
            "orientation": "LPS", "intensity": "native_after_nifti_scaling",
            "resize": False, "z_resampling": False, "hu_window": None,
            "normalization": None, "assume_mm": self.assume_mm,
            "patch_selection": "all spatial tiles; independent of ground truth",
        }
