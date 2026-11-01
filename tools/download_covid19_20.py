"""Download the public 199-case COVID-19-20 training split from Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "covid19_20"
REPO = "MedOtter/COVID-19-20"


def main() -> None:
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=DATA,
        allow_patterns=["Train/*.nii.gz", "README.md", "COVID-19-20_TrainValidation.xlsx"],
        max_workers=8,
    )
    ct_files = sorted((DATA / "Train").glob("*_ct.nii.gz"))
    mask_files = sorted((DATA / "Train").glob("*_seg.nii.gz"))
    if len(ct_files) != 199 or len(mask_files) != 199:
        raise RuntimeError(f"Expected 199 CT/mask pairs, found {len(ct_files)} CT and {len(mask_files)} masks")
    manifest = []
    for ct_path in ct_files:
        case_id = ct_path.name.removesuffix("_ct.nii.gz")
        mask_path = ct_path.with_name(f"{case_id}_seg.nii.gz")
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        ct, mask = nib.load(str(ct_path)), nib.load(str(mask_path))
        if ct.shape != mask.shape or not np.allclose(ct.affine, mask.affine):
            raise ValueError(f"CT/mask geometry differs for {case_id}")
        manifest.append({
            "id": case_id,
            "image": str(ct_path.relative_to(DATA)).replace("\\", "/"),
            "mask": str(mask_path.relative_to(DATA)).replace("\\", "/"),
            "shape": list(ct.shape),
            "spacing_mm": [float(value) for value in ct.header.get_zooms()[:3]],
        })
    (DATA / "cases.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    size = sum(path.stat().st_size for path in ct_files + mask_files)
    print(f"Verified {len(manifest)} aligned CT/mask pairs ({size / 1e9:.2f} GB compressed).")


if __name__ == "__main__":
    main()
