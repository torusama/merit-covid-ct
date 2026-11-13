"""Single-case pilot correctness; synthetic inputs only in automated tests."""
from copy import deepcopy
import hashlib
import json

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import load_development_split, validate_experiment
from tools.merit_real_case_trial import file_hash, run_trial, verified_case


@pytest.fixture
def pilot(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    config = validate_experiment(synthetic_experiment(tmp_path / "data"))
    cases, split, _, split_hash = load_development_split(config)
    part = split["folds"][0]
    case = next(c for c in cases if c.case_id in part["train"]["case_ids"])
    values = np.random.default_rng(13).uniform(0, 1, (32, 32, 13)).astype(np.float32)
    labels = np.zeros(values.shape, np.uint8)
    labels[0, 0, 6] = 1
    for path, array in ((case.image, values), (case.mask, labels)):
        image = nib.Nifti1Image(array, np.diag([-1., -1., 3., 1.]))
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    spec = {"case_id": case.case_id, "group_id": case.group_id, "fold": 0,
        "shape_xyz": list(values.shape), "split_sha256_lf": split_hash,
        "manifest_fingerprint": split["manifest_fingerprint"],
        "sha256": {"ct": file_hash(case.image), "gt": file_hash(case.mask)}}
    yield config_path, config["data"]["split_path"], spec, case, part
    torch.set_num_threads(old)


def test_real_case_pilot_native_tiny_lesion_update_and_ct_only_inference(pilot, tmp_path):
    config, split_path, spec, case, _ = pilot
    result = run_trial(config, case.image, case.mask, split_path, spec,
        tmp_path / "pilot.json", steps=1, device="cpu", check_equivalence=False)
    assert result["status"] == "PASS", result
    assert result["weights_changed"] and result["optimizer_steps"] == 1
    assert result["input_shape"] == [1, 1, 11, 32, 32]
    assert result["input_checks"][0]["positive_voxels_per_selected_slice"][5] == 1
    assert result["input_checks"][0]["presence_per_selected_slice"][5]
    assert result["source_files_unchanged"] and not result["xy_interpolation_called"]
    assert result["inference"]["ct_only_guard"] == "PASS"
    assert not result["inference"]["full_volume_reconstructed"]
    assert not result["checkpoint_saved"] and not result["accuracy_evaluated"]


def test_case_identity_rejects_wrong_gt_bytes(pilot):
    _, split_path, spec, case, _ = pilot
    changed = deepcopy(spec)
    changed["sha256"]["gt"] = "0" * 64
    with pytest.raises(ValueError, match="bytes do not match"):
        verified_case(case.image, case.mask, split_path, changed)


def test_case_identity_rejects_validation_case(pilot):
    _, split_path, spec, case, part = pilot
    changed = deepcopy(spec)
    changed["case_id"] = part["validation"]["case_ids"][0]
    changed["group_id"] = part["validation"]["group_ids"][0]
    with pytest.raises(ValueError, match="TRAIN partition"):
        verified_case(case.image, case.mask, split_path, changed)


def test_case_identity_rejects_modified_split(pilot, tmp_path):
    _, split_path, spec, case, _ = pilot
    other = tmp_path / "other_split.json"
    other.write_bytes(__import__("pathlib").Path(split_path).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="split bytes"):
        verified_case(case.image, case.mask, other, spec)
