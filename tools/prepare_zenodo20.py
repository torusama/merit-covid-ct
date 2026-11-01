"""Prepare the official 20-case Zenodo dataset under ``data/zenodo20``.

Run ``python tools/download_zenodo20.py`` first. This script is safe to rerun.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "archives" / "zenodo20"
DATA = ROOT / "data" / "zenodo20"


def case_id(filename: str) -> str | None:
    """Map the source filename to the stable identifier used in cases.json."""
    if filename.startswith("coronacases_"):
        match = re.search(r"(\d{3})\.nii(?:\.gz)?$", filename)
        return f"coronacases_{match.group(1)}" if match else None
    if filename in {"case009.nii", "case009.nii.gz"}:
        return "coronacases_009"
    if filename.startswith("radiopaedia_org_"):
        match = re.search(r"-(\d+_\d+_\d+)-dcm\.nii(?:\.gz)?$", filename)
        return f"radiopaedia_{match.group(1)}" if match else None
    if filename.startswith("radiopaedia_"):
        return filename.removesuffix(".gz").removesuffix(".nii")
    return None


def extract_nifti_files(archive_path: Path, destination: Path) -> None:
    """Extract NIfTI files without trusting archive directory paths."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            filename = Path(info.filename).name
            identifier = case_id(filename)
            if info.is_dir() or identifier is None:
                continue
            target = (destination / filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive path: {info.filename}")
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)


def indexed_nifti_files(directory: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in directory.iterdir():
        identifier = case_id(path.name)
        if identifier is None or not path.is_file():
            continue
        if identifier in indexed:
            raise ValueError(f"Duplicate case {identifier} in {directory}")
        indexed[identifier] = path
    return indexed


def main() -> None:
    sources = {
        "ct": ARCHIVES / "COVID-19-CT-Seg_20cases.zip",
        "combined": ARCHIVES / "Lung_and_Infection_Mask.zip",
        "infection": ARCHIVES / "Infection_Mask.zip",
        "lung": ARCHIVES / "Lung_Mask.zip",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing source archive(s):\n" + "\n".join(missing))

    extract_nifti_files(sources["ct"], DATA / "ct")
    for label in ("combined", "infection", "lung"):
        extract_nifti_files(sources[label], DATA / "masks" / label)

    images = indexed_nifti_files(DATA / "ct")
    combined = indexed_nifti_files(DATA / "masks" / "combined")
    infection = indexed_nifti_files(DATA / "masks" / "infection")
    lung = indexed_nifti_files(DATA / "masks" / "lung")
    expected = set(images)
    if len(expected) != 20 or any(group != expected for group in (combined, infection, lung)):
        raise ValueError(
            "Expected 20 matching CT, combined, infection, and lung-mask cases; "
            f"found CT={len(images)}, combined={len(combined)}, infection={len(infection)}, lung={len(lung)}"
        )

    cases = []
    for identifier in sorted(expected):
        image = nib.load(str(images[identifier]))
        masks = [nib.load(str(path)) for path in (combined[identifier], infection[identifier], lung[identifier])]
        if any(mask.shape != image.shape for mask in masks):
            raise ValueError(f"Shape mismatch in {identifier}")
        if any(not np.allclose(mask.affine, image.affine) for mask in masks):
            raise ValueError(f"Affine mismatch in {identifier}")
        cases.append({
            "id": identifier,
            "source": "Coronacases" if identifier.startswith("coronacases") else "Radiopaedia",
            "image": str(images[identifier].relative_to(DATA)).replace("\\", "/"),
            "mask": str(infection[identifier].relative_to(DATA)).replace("\\", "/"),
            "combined_mask": str(combined[identifier].relative_to(DATA)).replace("\\", "/"),
            "shape": list(image.shape),
            "spacing_mm": [float(value) for value in image.header.get_zooms()[:3]],
        })
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Verified {len(cases)} aligned CT/mask cases in {DATA}")


if __name__ == "__main__":
    main()
