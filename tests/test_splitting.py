from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from merit_covid.data.manifest import Case
from merit_covid.data.splitting import load_splits, make_splits, save_splits, validate_splits


def cases(count=20):
    return [Case("covid19_20", f"v{i:03d}", Path(f"/ct{i}.nii"), Path(f"/mask{i}.nii"),
                 patient_id=f"verified-patient-{i // 2}") for i in range(count)]


def test_reproducibility_grouping_and_coverage(tmp_path):
    records = cases()
    payload = make_splits(records, expected_count=20)
    assert payload == make_splits(list(reversed(records)), expected_count=20)
    assert payload != make_splits(records, seed=43, expected_count=20)
    for fold in payload["folds"]:
        groups = [{c.patient_id for c in records if c.case_id in fold[role]} for role in ("train", "validation", "internal_test")]
        assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
    path = tmp_path / "folds.json"
    save_splits(payload, path)
    assert load_splits(path, records) == payload
    with pytest.raises(FileExistsError):
        save_splits(payload, path)


def test_external_counts_and_duplicate_rejection():
    with pytest.raises(ValueError, match="Zenodo20"):
        make_splits([replace(c, dataset="zenodo20") for c in cases()], expected_count=20)
    with pytest.raises(ValueError, match="Expected 199"):
        make_splits(cases())
    with pytest.raises(ValueError, match="Duplicate"):
        make_splits(cases() + [cases()[0]], expected_count=21)


def test_tampered_split_leakage_and_metadata():
    records = cases()
    payload = make_splits(records, expected_count=20)
    bad = deepcopy(payload)
    bad["folds"][0]["train"].append(bad["folds"][0]["validation"][0])
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(bad, records)
    bad = deepcopy(payload)
    # Swap one of a patient's volumes between train and validation without ID overlap.
    train, val = bad["folds"][0]["train"], bad["folds"][0]["validation"]
    train[0], val[0] = val[0], train[0]
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(bad, records)
    with pytest.raises(ValueError, match="metadata"):
        validate_splits(payload, [replace(records[0], patient_id="changed")] + records[1:])
