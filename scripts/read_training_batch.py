"""Inspect one saved-fold CPU batch without constructing a model or training.

Run from any working directory; default paths are relative to this repository.
This maintained CLI replaces the deleted examples/read_training_batch.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from merit_covid import PreprocessConfig, build_fold_loaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/covid19_20/cases.json")
    parser.add_argument("--split", type=Path, default=ROOT / "splits/covid19_20_5fold.json")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, nargs=2, help="Explicit H W resize; omit to preserve native XY")
    parser.add_argument("--slab-size", type=int, choices=(1, 3, 5, 11), default=11)
    parser.add_argument("--assume-mm", action="store_true")
    parser.add_argument("--patient-map", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    preprocess = PreprocessConfig(slab_size=args.slab_size,
        size_hw=tuple(args.size) if args.size else None,
        intensity_encoding="native", assume_mm=args.assume_mm)
    loaders = build_fold_loaders(manifest=args.manifest, split_path=args.split,
        fold=args.fold, preprocess=preprocess, batch_size=args.batch_size,
        shuffle_train=False, seed=args.seed, num_workers=args.workers,
        patient_map=args.patient_map)
    batch = next(iter(getattr(loaders, args.partition)))
    inputs, targets, offsets = batch["image_slab"], batch["master_mask"], batch["z_offsets_mm"]
    report = {
        "mode": "inspect_batch_only", "device": str(inputs.device),
        "fold": args.fold, "selected_partition": args.partition,
        "volumes_in_partition": {name: len(getattr(loaders, name).dataset.cases)
                                  for name in ("train", "validation", "test")},
        "master_samples_in_partition": {name: len(getattr(loaders, name).dataset)
                                         for name in ("train", "validation", "test")},
        "inputs": {"shape": list(inputs.shape), "dtype": str(inputs.dtype),
                   "value_range": [float(inputs.min()), float(inputs.max())]},
        "targets": {"shape": list(targets.shape), "dtype": str(targets.dtype),
                    "labels": torch.unique(targets).tolist()},
        "z_offsets_mm_shape": list(offsets.shape), "case_ids": batch["case_id"],
        "master_indices": batch["master_slice_index"].tolist(),
        "first_sample_offsets_mm": offsets[0].tolist(),
        "first_sample_original_shape_xyz": batch["metadata"][0]["original_shape_xyz"],
        "first_sample_orientation": batch["metadata"][0]["canonical_orientation"],
        "first_sample_processed_size_hw": batch["metadata"][0]["processed_size_hw"],
        "note": "Patient-level independence remains unverified without trusted patient IDs.",
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
