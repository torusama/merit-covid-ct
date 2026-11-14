"""Fit native-scaled voxel z-score on a saved fold's full train CT partition only.

Output is a new complete experiment YAML with absolute data paths. No GT,
validation or test voxels are opened, and no input file is modified.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from merit_covid.engine.training import load_experiment, load_development_split


def fit_train_zscore(config, fold, block_depth=8):
    if type(fold) is not int or fold not in range(5) or type(block_depth) is not int or block_depth < 1:
        raise ValueError("fold must be 0..4 and block_depth positive")
    if config["preprocess"]["intensity_encoding"] != "native":
        raise ValueError("Fit requires a native-intensity source configuration")
    cases, split, _, split_hash = load_development_split(config)
    selected = set(next(f for f in split["folds"] if f["fold"] == fold)["train"]["case_ids"])
    count, mean, m2 = 0, 0.0, 0.0
    for case in cases:
        if case.case_id not in selected:
            continue
        image = nib.load(str(case.image))
        if len(image.shape) != 3:
            raise ValueError(f"{case.case_id}: expected 3D CT")
        for z in range(0, image.shape[2], block_depth):
            block = np.asarray(image.dataobj[:, :, z:z + block_depth], dtype=np.float32)
            if not block.size or not np.isfinite(block).all():
                raise ValueError(f"{case.case_id}: non-finite or empty CT")
            n = block.size
            block_mean = float(np.mean(block, dtype=np.float64))
            block_m2 = float(np.var(block, dtype=np.float64)) * n
            total = count + n
            delta = block_mean - mean
            m2 += block_m2 + delta * delta * count * n / total
            mean += delta * n / total
            count = total
    if not count or not np.isfinite(m2) or m2 / count <= 1e-16:
        raise ValueError("Training voxels have invalid or near-zero variance")
    return {"zscore_mean": mean, "zscore_std": float(np.sqrt(m2 / count)),
            "zscore_fit": {"fold": fold, "split_sha256_lf": split_hash,
                           "case_ids": sorted(selected), "method": "native_scaled_voxel_population"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_experiment(args.config)
    fit = fit_train_zscore(config, args.fold)
    config["preprocess"].update(intensity_encoding="train_zscore", **fit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


if __name__ == "__main__":
    main()
