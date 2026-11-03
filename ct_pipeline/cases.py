"""Manifest validation and explicit patient identity provenance."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    case_id: str
    image: Path
    mask: Path
    group_id: str


def identifier(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def load_cases(manifest, expected_count=None, patient_map=None):
    """Paths are relative to the manifest. A volume ID is never a patient ID."""
    manifest = Path(manifest).resolve()
    records = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a nonempty list of cases")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} cases, found {len(records)}")
    ids = [identifier(r["id"], "id") for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs")
    if patient_map is not None:
        mapping = json.loads(Path(patient_map).read_text(encoding="utf-8-sig"))
        if not isinstance(mapping, dict) or set(mapping) != set(ids):
            raise ValueError("Patient map must map exactly every manifest case ID")
        source = "user_supplied_patient_map"
        patient_ids_available = True
    else:
        fields = [key for key in ("patient_id", "group_id") if any(key in r for r in records)]
        key = fields[0] if fields else None
        if key and not all(key in r for r in records):
            raise ValueError(f"Incomplete {key}: do not mix patient/group IDs with volume IDs")
        mapping = {r["id"]: r[key] if key else r["id"] for r in records}
        source = f"manifest_{key}" if key else "volume_id_fallback"
        patient_ids_available = key == "patient_id"
    cases, paths = [], set()
    for record in records:
        case_id = record["id"]
        pair = []
        for key in ("image", "mask"):
            path = (manifest.parent / identifier(record[key], key)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{case_id}: missing {key}: {path}")
            if path in paths:
                raise ValueError(f"Reused CT/mask path in manifest: {path}")
            paths.add(path)
            pair.append(path)
        cases.append(Case(case_id, *pair, identifier(mapping[case_id], "group_id")))
    cases.sort(key=lambda c: c.case_id)
    provenance = {
        "group_source": source,
        "patient_ids_available": patient_ids_available,
        "identity_note": (
            "Grouping respects supplied patient IDs; their clinical identity must be verified upstream."
            if patient_ids_available else
            "Patient identity is unverified. Group/volume independence does not establish patient independence across series."
        ),
    }
    return cases, provenance


def manifest_fingerprint(cases):
    # Content identity, independent of absolute clone location.
    payload = [(c.case_id, c.group_id, c.image.name, c.mask.name) for c in sorted(cases, key=lambda c: c.case_id)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
