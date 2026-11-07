from dataclasses import replace
import json

import pytest

from merit_covid.data.manifest import Case, legacy_mask_path, load_manifest, validate_cases, write_manifest


def test_versioned_roundtrip_and_no_overwrite(nifti_case, tmp_path):
    case, _, _ = nifti_case()
    target = tmp_path / "manifest.json"
    write_manifest([case], target)
    assert load_manifest(target) == [case]
    assert not json.loads(target.read_text())["cases"][0]["image"].startswith(str(tmp_path))
    with pytest.raises(FileExistsError):
        write_manifest([case], target)


@pytest.mark.parametrize("dataset", ["covid19_20", "zenodo20"])
def test_legacy_manifest(nifti_case, tmp_path, dataset):
    case, _, _ = nifti_case(dataset=dataset)
    target = tmp_path / "cases.json"
    target.write_text(json.dumps([{"id": case.case_id, "image": str(case.image), "mask": str(case.infection_mask)}]))
    loaded = load_manifest(target, dataset=dataset)[0]
    assert loaded.patient_id is None
    assert loaded.intensity == "unknown"
    assert loaded.infection_mask == case.infection_mask


def test_duplicate_ids_paths_missing_and_invalid(nifti_case):
    case, _, _ = nifti_case()
    with pytest.raises(ValueError, match="Duplicate case"):
        validate_cases([case, case])
    with pytest.raises(ValueError, match="Duplicate image"):
        validate_cases([case, replace(case, case_id="other")])
    with pytest.raises(FileNotFoundError):
        validate_cases([replace(case, image=case.image.with_name("absent.nii"))])
    with pytest.raises(ValueError, match="patient_id"):
        replace(case, patient_id="")
    with pytest.raises(ValueError, match="spacing"):
        replace(case, spacing_mm=(1, float("nan"), 5))
    with pytest.raises(ValueError, match="orientation"):
        replace(case, orientation=("R", "L", "S"))


def test_source_filename_resolution(tmp_path):
    directory = tmp_path / "masks" / "lung"
    directory.mkdir(parents=True)
    actual = directory / "coronacases_org_001.nii"
    actual.touch()
    row = {"id": "coronacases_001"}
    assert legacy_mask_path(row, tmp_path, "lung_mask") == actual
    row["lung_mask"] = "masks/lung/coronacases_org_001.nii"
    assert legacy_mask_path(row, tmp_path, "lung_mask") == actual
    with pytest.raises(FileNotFoundError):
        legacy_mask_path({"id": "x", "lung_mask": "missing.nii"}, tmp_path, "lung_mask")
    (directory / "coronacases_001.nii.gz").touch()
    with pytest.raises(ValueError, match="exactly one"):
        legacy_mask_path({"id": "coronacases_001"}, tmp_path, "lung_mask")


@pytest.mark.parametrize("payload", [{"schema_version": 2}, {"schema_version": 1, "cases": [1]}, [{"id": "x"}]])
def test_malformed_manifest(tmp_path, payload):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_manifest(path)
