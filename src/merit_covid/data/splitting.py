"""Reproducible grouped outer test folds with separate rotating validation folds."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .manifest import Case, validate_cases


def _group(case: Case) -> str:
    return f"patient:{case.patient_id}" if case.patient_id is not None else f"volume:{case.case_id}"


def manifest_fingerprint(cases: list[Case]) -> str:
    # Exclude machine-specific paths; retain identity and grouping evidence.
    records = sorted((c.dataset, c.case_id, c.patient_id) for c in cases)
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def make_splits(cases: list[Case], *, seed: int = 42, num_folds: int = 5,
                expected_count: int | None = 199) -> dict:
    validate_cases(cases, check_files=False)
    if any(c.dataset != "covid19_20" for c in cases):
        raise ValueError("Only covid19_20 may enter development splits; Zenodo20 is external only")
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} cases, found {len(cases)}")
    if num_folds < 3:
        raise ValueError("At least three folds required for disjoint train/validation/internal_test")
    groups = {}
    for case in sorted(cases, key=lambda c: c.case_id):
        groups.setdefault(_group(case), []).append(case.case_id)
    if len(groups) < num_folds:
        raise ValueError("Too few distinct patient/volume groups for requested folds")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    # Stable size sorting keeps random tie order and balances patient-group sizes.
    keys.sort(key=lambda k: len(groups[k]), reverse=True)
    buckets = [[] for _ in range(num_folds)]
    for key in keys:
        target = min(range(num_folds), key=lambda i: len(buckets[i]))
        buckets[target].extend(groups[key])
    folds = []
    for index in range(num_folds):
        val = (index + 1) % num_folds
        folds.append({
            "fold": index,
            "train": sorted(c for i, bucket in enumerate(buckets) if i not in (index, val) for c in bucket),
            "validation": sorted(buckets[val]),
            "internal_test": sorted(buckets[index]),
        })
    result = {"schema_version": 1, "dataset": "covid19_20", "seed": seed,
              "num_folds": num_folds, "case_count": len(cases),
              "manifest_fingerprint": manifest_fingerprint(cases),
              "grouping": "verified patient_id when present; otherwise volume case_id",
              "folds": folds}
    validate_splits(result, cases)
    return result


def validate_splits(payload: dict, cases: list[Case]) -> None:
    validate_cases(cases, check_files=False)
    if any(c.dataset != "covid19_20" for c in cases) or payload.get("dataset") != "covid19_20":
        raise ValueError("External datasets cannot enter development splits")
    by_id = {c.case_id: c for c in cases}
    expected = set(by_id)
    folds = payload.get("folds", [])
    if payload.get("schema_version") != 1 or payload.get("case_count") != len(cases) or payload.get("manifest_fingerprint") != manifest_fingerprint(cases):
        raise ValueError("Split metadata does not match the manifest")
    if len(folds) != payload.get("num_folds") or len(folds) < 3:
        raise ValueError("Split fold count mismatch")
    tests = Counter()
    for index, fold in enumerate(folds):
        if fold.get("fold") != index:
            raise ValueError("Duplicate or out-of-order fold identifiers")
        sets, patient_groups = [], []
        for role in ("train", "validation", "internal_test"):
            ids = fold.get(role, [])
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= expected:
                raise ValueError(f"Invalid, empty or duplicate case IDs in {role}")
            sets.append(set(ids))
            patient_groups.append({_group(by_id[c]) for c in ids})
        if set.union(*sets) != expected:
            raise ValueError("Each fold must cover the entire primary manifest")
        for i in range(3):
            for j in range(i):
                if sets[i] & sets[j] or patient_groups[i] & patient_groups[j]:
                    raise ValueError("Patient/volume leakage between split roles")
        tests.update(fold["internal_test"])
    if tests != Counter({c: 1 for c in expected}):
        raise ValueError("Each volume must occur exactly once in outer internal_test folds")


def save_splits(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def load_splits(path: str | Path, cases: list[Case]) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_splits(result, cases)
    return result
