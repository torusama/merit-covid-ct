"""Validated YAML configuration with paths relative to the configuration file."""

from dataclasses import dataclass
from pathlib import Path
import math

import yaml


@dataclass(frozen=True)
class Config:
    dataset_root: Path = Path("data/covid19_20")
    output_dir: Path = Path("outputs")
    seed: int = 42
    num_folds: int = 5
    num_slices: int = 11
    master_index: int = 5
    image_size: tuple[int, int] | None = None
    orientation: str = "RAS"
    normalization: str = "none"
    hu_window: tuple[float, float] = (-1000.0, 400.0)
    boundary: str = "edge"
    batch_size: int = 1
    num_workers: int = 0
    epochs: int = 100
    learning_rate: float = 0.0001
    device: str = "cpu"

    def __post_init__(self):
        for name in ("seed", "num_folds", "num_slices", "master_index", "batch_size", "num_workers", "epochs"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.num_slices not in (1, 3, 5, 11) or self.master_index != self.num_slices // 2:
            raise ValueError("Use 1/3/5/11 slices and a centered master_index")
        if self.num_folds < 3 or self.batch_size < 1 or self.num_workers < 0 or self.epochs < 1:
            raise ValueError("Invalid fold, batch, worker or epoch count")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.image_size is not None and (len(self.image_size) != 2 or any(type(v) is not int or v < 1 for v in self.image_size)):
            raise ValueError("image_size must be null or two positive integers")
        if self.orientation != "RAS":
            raise ValueError("The foundation supports RAS orientation only")
        if self.normalization not in ("none", "zscore", "hu_window", "windowed_255"):
            raise ValueError("Unknown normalization policy")
        if len(self.hu_window) != 2 or not all(math.isfinite(v) for v in self.hu_window) or self.hu_window[0] >= self.hu_window[1]:
            raise ValueError("hu_window requires finite increasing bounds")
        if self.boundary not in ("edge", "reflect"):
            raise ValueError("boundary must be edge or reflect")
        if self.device not in ("cpu", "auto", "cuda", "mps") and not self.device.startswith("cuda:"):
            raise ValueError("Unknown device specification")


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    defaults = Config()
    for key in ("dataset_root", "output_dir"):
        values[key] = (path.parent / values.get(key, getattr(defaults, key))).resolve()
    for key in ("image_size", "hu_window"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return Config(**values)
