"""Prepare the 20-case Zenodo COVID CT dataset under data/zenodo20.

The raw archives must be in archives/zenodo20/. The operation is resumable.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import nibabel as nib


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "archives" / "zenodo20"
DATA = ROOT / "data" / "zenodo20"


def case_id(filename: str) -> str | None:
    if filename.startswith("coronacases_"):
        match = re.search(r"(\d{3})\.nii$", filename)
        return f"coronacases_{match.group(1)}" if match else None
    if filename == "case009.nii":
        return "coronacases_009"
    if filename.startswith("radiopaedia_org_"):
        match = re.search(r"-(\d+_\d+_\d+)-dcm\.nii$", filename)
        return f"radiopaedia_{match.group(1)}" if match else None
    if filename.startswith("radiopaedia_"):
        return filename.removesuffix(".nii")
    return None


def extract_zip(archive_path: Path, destination: Path, *, keep_top_folder: str | None = None) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if keep_top_folder is not None:
                if not parts or parts[0] != keep_top_folder:
                    continue
                relative = Path(*parts[1:])
            else:
                relative = Path(info.filename)
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive path: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != info.file_size:
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)


def main() -> None:
    raw_ct = ARCHIVES / "kaggle_covid19_ct.zip"
    if not raw_ct.is_file():
        raise SystemExit(f"Missing source archive: {raw_ct}")
    DATA.mkdir(parents=True, exist_ok=True)
    extract_zip(raw_ct, DATA / "ct", keep_top_folder="COVID-19-CT-Seg_20cases")
    extract_zip(raw_ct, DATA / "masks" / "combined", keep_top_folder="Lung_and_Infection_Mask")
    for label in ("Infection_Mask", "Lung_Mask"):
        archive = ARCHIVES / f"{label}.zip"
        if not archive.is_file():
            raise SystemExit(f"Missing source archive: {archive}")
        extract_zip(archive, DATA / "masks" / label.removesuffix("_Mask").lower())

    ct_dir = DATA / "ct"
    combined_dir = DATA / "masks" / "combined"
    images = {case_id(path.name): path for path in ct_dir.glob("*.nii") if case_id(path.name)}
    combined = {case_id(path.name): path for path in combined_dir.glob("*.nii") if case_id(path.name)}
    if set(images) != set(combined) or len(images) != 20:
        raise ValueError(f"Expected 20 matched CT/combined-mask cases; CT={len(images)}, masks={len(combined)}")

    cases = []
    for identifier in sorted(images):
        image = nib.load(str(images[identifier]))
        mask = nib.load(str(combined[identifier]))
        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch in {identifier}: {image.shape} vs {mask.shape}")
        infection = DATA / "masks" / "infection" / f"{identifier}.nii.gz"
        lung = DATA / "masks" / "lung" / f"{identifier}.nii.gz"
        if not infection.is_file() or not lung.is_file():
            raise FileNotFoundError(f"Missing original mask for {identifier}")
        cases.append({
            "id": identifier,
            "source": "Coronacases" if identifier.startswith("coronacases") else "Radiopaedia",
            "image": str(images[identifier].relative_to(DATA)).replace("\\", "/"),
            "mask": str(infection.relative_to(DATA)).replace("\\", "/"),
            "combined_mask": str(combined[identifier].relative_to(DATA)).replace("\\", "/"),
            "shape": list(image.shape),
            "spacing_mm": [float(value) for value in image.header.get_zooms()[:3]],
        })
    (DATA / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Verified {len(cases)} aligned CT/mask cases in {DATA}")


if __name__ == "__main__":
    main()
