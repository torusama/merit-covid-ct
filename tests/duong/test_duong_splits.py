from collections import Counter
from dataclasses import replace
import copy
import json

import pytest

from ct_pipeline.cases import Case, load_cases
from ct_pipeline.splits import create_splits, validate_splits


def fixture_cases(tmp_path):
    return [Case(f"case{i:02d}", tmp_path / f"ct{i}", tmp_path / f"mask{i}", f"patient{i // 2}") for i in range(23)]


def test_group_disjoint_reproducible_and_complete(tmp_path):
    cases = fixture_cases(tmp_path)
    split = create_splits(cases, {"group_source": "synthetic_test_only"})
    assert split == create_splits(list(reversed(cases)), {"group_source": "synthetic_test_only"})
    assert split != create_splits(cases, {"group_source": "synthetic_test_only"}, seed=43)
    test_ids = Counter()
    for fold in split["folds"]:
        groups = [set(fold[p]["group_ids"]) for p in ("train", "validation", "test")]
        assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        assert set().union(*(set(fold[p]["case_ids"]) for p in ("train", "validation", "test"))) == {c.case_id for c in cases}
        test_ids.update(fold["test"]["case_ids"])
    assert test_ids == Counter({c.case_id: 1 for c in cases})


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "unknown", "group", "fingerprint", "test_repeat"])
def test_reject_corrupted_split(tmp_path, corruption):
    cases = fixture_cases(tmp_path)
    split = create_splits(cases, {})
    part = split["folds"][0]["train"]
    if corruption == "duplicate":
        part["case_ids"].append(part["case_ids"][0])
    elif corruption == "missing":
        part["case_ids"].pop()
    elif corruption == "unknown":
        part["case_ids"][0] = "unknown"
    elif corruption == "group":
        part["group_ids"] = ["wrong"]
    elif corruption == "fingerprint":
        cases[0] = replace(cases[0], group_id="changed")
    else:
        split["folds"][1] = {**copy.deepcopy(split["folds"][0]), "fold": 1}
    with pytest.raises(ValueError):
        validate_splits(split, cases)


def test_manifest_identity_and_validation(tmp_path, write_case):
    first, second = write_case("a"), write_case("b")
    records = [{"id": c.case_id, "image": c.image.name, "mask": c.mask.name} for c in (first, second)]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(records))
    cases, identity = load_cases(path, 2)
    assert identity["group_source"] == "volume_id_fallback"
    assert not identity["patient_ids_available"]
    records[0]["patient_id"] = "p"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Incomplete"):
        load_cases(path)
    records[1]["patient_id"] = "p"
    path.write_text(json.dumps(records))
    cases, identity = load_cases(path)
    assert cases[0].group_id == cases[1].group_id == "p"
    assert identity["patient_ids_available"]
    with pytest.raises(ValueError, match="Expected 199"):
        load_cases(path, 199)
    records[1]["id"] = "a"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Duplicate"):
        load_cases(path)


def test_missing_reused_paths_and_patient_map(tmp_path, write_case):
    a, b = write_case("a"), write_case("b")
    records = [{"id": c.case_id, "image": c.image.name, "mask": c.mask.name} for c in (a, b)]
    path, mapping = tmp_path / "cases.json", tmp_path / "patients.json"
    path.write_text(json.dumps(records))
    mapping.write_text(json.dumps({"a": "p1", "b": "p1"}))
    cases, identity = load_cases(path, patient_map=mapping)
    assert cases[0].group_id == cases[1].group_id
    assert identity["group_source"] == "user_supplied_patient_map"
    mapping.write_text(json.dumps({"a": "p1"}))
    with pytest.raises(ValueError, match="exactly"):
        load_cases(path, patient_map=mapping)
    records[1]["image"] = a.image.name
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Reused"):
        load_cases(path)
    records[1]["image"] = "missing.nii.gz"
    path.write_text(json.dumps(records))
    with pytest.raises(FileNotFoundError):
        load_cases(path)
