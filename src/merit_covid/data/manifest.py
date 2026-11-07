"""Unified manifests and a non-destructive reader for both legacy cases.json files."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import numpy as np


DATASETS = {"covid19_20", "zenodo20"}


def normalized_zenodo_id(filename: str) -> str | None:
    if filename.startswith("coronacases_"):
        match = re.search(r"(\d{3})\.nii(?:\.gz)?$", filename)
        return f"coronacases_{match.group(1)}" if match else None
    if filename in {"case009.nii", "case009.nii.gz"}:
        return "coronacases_009"
    if filename.startswith("radiopaedia_org_"):
        match = re.search(r"-(\d+_\d+_\d+)-dcm\.nii(?:\.gz)?$", filename)
        return f"radiopaedia_{match.group(1)}" if match else None
    if filename.startswith("radiopaedia_") and filename.endswith((".nii", ".nii.gz")):
        return filename.removesuffix(".gz").removesuffix(".nii")
    return None


def legacy_mask_path(case: dict, root: Path, key: str) -> Path:
    """Prefer recorded paths; index actual source names only for omitted lung paths."""
    if case.get(key):
        path = root / case[key]
        if not path.is_file():
            raise FileNotFoundError(f"{case['id']}: missing {key}: {path}")
        return path
    if key != "lung_mask":
        raise ValueError(f"{case['id']}: missing explicit {key} path")
    directory = root / "masks" / "lung"
    matches = [p for p in directory.glob("*.nii*") if normalized_zenodo_id(p.name) == case["id"]]
    if len(matches) != 1:
        raise ValueError(f"{case['id']}: expected exactly one lung mask in {directory}; found {len(matches)}")
    return matches[0]


@dataclass(frozen=True)
class Case:
    dataset: str
    case_id: str
    image: Path
    infection_mask: Path | None
    source: str = "unknown"
    patient_id: str | None = None
    lung_mask: Path | None = None
    shape: tuple[int, int, int] | None = None
    spacing_mm: tuple[float, float, float] | None = None
    affine: tuple[tuple[float, ...], ...] | None = None
    orientation: tuple[str, str, str] | None = None
    intensity: str = "unknown"
    group_id: str | None = None

    def __post_init__(self):
        if self.dataset not in DATASETS or not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("A supported dataset and non-empty case_id are required")
        if self.patient_id is not None and (not isinstance(self.patient_id, str) or not self.patient_id.strip()):
            raise ValueError(f"{self.case_id}: patient_id must be a verified non-empty string or null")
        if self.group_id is not None and (not isinstance(self.group_id, str) or not self.group_id.strip()):
            raise ValueError(f"{self.case_id}: group_id must be a non-empty string or null")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError(f"{self.case_id}: source must be a non-empty string")
        if self.intensity not in ("unknown", "HU", "windowed_255"):
            raise ValueError(f"{self.case_id}: unsupported intensity provenance")
        if self.shape is not None and (len(self.shape) != 3 or any(type(v) is not int or v < 1 for v in self.shape)):
            raise ValueError(f"{self.case_id}: invalid 3D shape")
        if self.spacing_mm is not None and (len(self.spacing_mm) != 3 or not np.all(np.isfinite(self.spacing_mm)) or min(self.spacing_mm) <= 0):
            raise ValueError(f"{self.case_id}: invalid spacing_mm")
        if self.affine is not None:
            affine = np.asarray(self.affine)
            if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1]) or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
                raise ValueError(f"{self.case_id}: invalid affine")
        if self.orientation is not None:
            axis = {"L": 0, "R": 0, "A": 1, "P": 1, "S": 2, "I": 2}
            if len(self.orientation) != 3 or {axis.get(code) for code in self.orientation} != {0, 1, 2}:
                raise ValueError(f"{self.case_id}: invalid orientation")


def validate_cases(cases: list[Case], *, check_files: bool = True) -> None:
    if not cases:
        raise ValueError("Manifest contains no cases")
    ids, images, masks = set(), set(), set()
    for case in cases:
        key = (case.dataset, case.case_id)
        if key in ids:
            raise ValueError(f"Duplicate case ID: {key}")
        ids.add(key)
        for value, seen, kind in ((case.image, images, "image"), (case.infection_mask, masks, "infection mask")):
            if value is not None:
                path = Path(value).resolve()
                if path in seen:
                    raise ValueError(f"Duplicate {kind} path: {path}")
                seen.add(path)
        if check_files:
            for value in (case.image, case.infection_mask, case.lung_mask):
                if value is not None and not Path(value).is_file():
                    raise FileNotFoundError(f"{case.case_id}: {value}")


def load_manifest(path: str | Path, *, dataset: str | None = None, check_files: bool = True) -> list[Case]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
        dataset = dataset or path.parent.name
        legacy = True
    elif isinstance(payload, dict) and payload.get("schema_version") == 1:
        records = payload.get("cases")
        legacy = False
    else:
        raise ValueError("Expected legacy list or schema_version=1 manifest")
    if not isinstance(records, list):
        raise ValueError("Manifest cases must be a list")
    cases = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest record {index} must be a mapping")
        try:
            image = row["image"]
            if not isinstance(image, str) or not image.strip():
                raise ValueError("image must be a non-empty path string")
            mask = row.get("mask") if legacy else row.get("infection_mask")
            values = dict(
                dataset=dataset if legacy else row["dataset"],
                case_id=row["id"] if legacy else row["case_id"],
                image=(path.parent / image).resolve(),
                infection_mask=(path.parent / mask).resolve() if mask else None,
                source=row.get("source", "unknown"), patient_id=row.get("patient_id"),
                lung_mask=(path.parent / row["lung_mask"]).resolve() if row.get("lung_mask") else None,
                shape=tuple(row["shape"]) if row.get("shape") is not None else None,
                spacing_mm=tuple(row["spacing_mm"]) if row.get("spacing_mm") is not None else None,
                affine=tuple(tuple(r) for r in row["affine"]) if row.get("affine") is not None else None,
                orientation=tuple(row["orientation"]) if row.get("orientation") is not None else None,
                intensity=row.get("intensity", "unknown"),
                group_id=row.get("group_id"),
            )
            cases.append(Case(**values))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid manifest record {index}: {exc}") from exc
    validate_cases(cases, check_files=check_files)
    if cases and all(case.dataset == "covid19_20" for case in cases):
        for field_name in ("patient_id", "group_id"):
            supplied = [getattr(case, field_name) is not None for case in cases]
            if any(supplied) and not all(supplied):
                raise ValueError(f"Incomplete {field_name}: do not mix patient/group identities with volume IDs")
    return cases


def write_manifest(cases: list[Case], path: str | Path) -> None:
    """Write portable relative paths; exclusive creation protects existing manifests."""
    import os
    validate_cases(cases, check_files=False)
    path = Path(path).resolve()
    rows = []
    for case in cases:
        row = asdict(case)
        for key in ("image", "infection_mask", "lung_mask"):
            if row[key] is not None:
                row[key] = Path(os.path.relpath(row[key], path.parent)).as_posix()
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "cases": rows}, handle, indent=2, allow_nan=False)
