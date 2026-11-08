"""Small, real NIfTI software fixtures for CLI smoke; not benchmark data."""

from copy import deepcopy
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from merit_covid.data.pipeline.cases import load_cases
from merit_covid.data.pipeline.splits import create_splits


def synthetic_experiment(root, *, device="cpu", sampling_mode="index", target_spacing_mm=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(10):
        values = np.random.default_rng(i).uniform(0, 1, (16, 18, 2)).astype(np.float32)
        mask = np.zeros(values.shape, dtype=np.uint8)
        mask[3:7, 5:9, i % 2] = 1
        for label, data in (("ct", values), ("mask", mask)):
            path = root / f"c{i}_{label}.nii.gz"
            if path.exists():
                raise FileExistsError(path)
            image = nib.Nifti1Image(data, np.diag([-0.7, -0.8, (1, 1.5, 3, 5)[i % 4], 1]))
            image.header.set_xyzt_units("mm")
            nib.save(image, path)
        records.append({"id": f"c{i}", "image": f"c{i}_ct.nii.gz", "mask": f"c{i}_mask.nii.gz", "patient_id": f"synthetic-p{i}"})
    manifest = root / "cases.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    cases, identity = load_cases(manifest)
    splits = create_splits(cases, identity)
    split_path = root / "split.json"
    split_path.write_text(json.dumps(splits), encoding="utf-8")
    config = {"data": {"manifest": str(manifest.resolve()), "split_path": str(split_path.resolve())},
        "preprocess": {"slab_size": 11, "size_hw": [32, 32], "intensity_encoding": "native",
                       "sampling_mode": sampling_mode, "target_spacing_mm": target_spacing_mm},
        "model": {"img_size": [11, 32, 32], "feature_size": 2, "depths": [1, 1, 1, 1],
                  "position_mode": "index"}, "loss": {},
        "training": {"epochs": 1, "batch_size": 2, "num_workers": 0, "device": device,
                     "include_hd95": False, "seed": 7}}
    return config
