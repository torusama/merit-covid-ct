"""Reproducible grouped outer test folds with separate rotating validation folds."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .manifest import Case, validate_cases


def _group(case: Case) -> str:
    return f"patient:{case.patient_id}" if case.patient_id is not None else f"group:{case.group_id}" if case.group_id is not None else f"volume:{case.case_id}"


def manifest_fingerprint(cases: list[Case]) -> str:
    # Exclude machine-specific paths; retain identity and grouping evidence.
    records = sorted((c.dataset, c.case_id, c.patient_id, c.group_id) for c in cases)
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def make_splits(cases: list[Case], *, seed: int = 42, num_folds: int = 5,
                expected_count: int | None = 199) -> dict:
    validate_cases(cases, check_files=False)
    if any(c.dataset != "covid19_20" for c in cases):
        raise ValueError("Only covid19_20 may enter development splits; Zenodo20 is external only")
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} cases, found {len(cases)}")
    if num_folds!=5:
        raise ValueError('The agreed Duong protocol requires five folds')
    from .pipeline.cases import Case as PipelineCase
    from .pipeline.splits import create_splits
    linked=[PipelineCase(c.case_id,c.image,c.infection_mask,c.patient_id or c.group_id or c.case_id) for c in cases]
    identity={'patient_ids_available':all(c.patient_id is not None for c in cases)}
    canonical=create_splits(linked,identity,seed=seed,validation_fraction=0.2)
    return _main_view(canonical,cases)

def _main_view(canonical,cases):
    result={'schema_version':1,'dataset':'covid19_20','seed':canonical['seed'],'num_folds':5,'case_count':len(cases),
            'manifest_fingerprint':manifest_fingerprint(cases),'_canonical':canonical,
            'folds':[{'fold':f['fold'],'train':list(f['train']['case_ids']),'validation':list(f['validation']['case_ids']),
                      'internal_test':list(f['test']['case_ids'])} for f in canonical['folds']]}
    validate_splits(result,cases)
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
    if "_canonical" in payload:
        from .pipeline.cases import Case as PipelineCase
        from .pipeline.splits import validate_splits as validate_canonical
        linked = [PipelineCase(c.case_id, c.image, c.infection_mask, c.patient_id or c.group_id or c.case_id) for c in cases]
        validate_canonical(payload["_canonical"], linked)
        for view, source in zip(folds, payload["_canonical"]["folds"], strict=True):
            if any(view[key] != source[role]["case_ids"] for key, role in (("train", "train"), ("validation", "validation"), ("internal_test", "test"))):
                raise ValueError("Split compatibility view differs from canonical data")


def save_splits(payload: dict, path: str | Path) -> None:
    canonical = payload["_canonical"]
    for view, source in zip(payload["folds"], canonical["folds"], strict=True):
        if any(view[key] != source[role]["case_ids"] for key, role in (("train", "train"), ("validation", "validation"), ("internal_test", "test"))):
            raise ValueError("Split compatibility view differs from canonical data")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload["_canonical"], handle, indent=2, allow_nan=False)


def load_splits(path: str | Path, cases: list[Case]) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    from .pipeline.cases import Case as PipelineCase
    from .pipeline.splits import validate_splits as validate_canonical
    linked=[PipelineCase(c.case_id,c.image,c.infection_mask,c.patient_id or c.group_id or c.case_id) for c in cases]
    validate_canonical(result,linked)
    return _main_view(result,cases)
