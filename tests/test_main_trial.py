"""Integrated-main notebook worker through real production Trainer APIs."""

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import load_development_split, validate_experiment
from tools.merit_main_trial import run_trial, file_hash


def fixture_case(tmp_path, device="cpu"):
    config = synthetic_experiment(tmp_path / "data", device=device)
    for path in (tmp_path / "data").glob("*.nii.gz"):
        image = nib.load(path)
        values = np.pad(image.get_fdata(dtype=np.float32), ((0, 16), (0, 14), (0, 0)))
        nib.save(nib.Nifti1Image(values, image.affine, image.header.copy()), path)
    config["preprocess"]["size_hw"] = None
    config["training"]["activation_checkpointing"] = True
    config = validate_experiment(config)
    cases, split, _, split_hash = load_development_split(config)
    case = next(c for c in cases if c.case_id in split["folds"][0]["train"]["case_ids"])
    spec = dict(case_id=case.case_id, group_id=case.group_id, fold=0,
        split_sha256_lf=split_hash, manifest_fingerprint=split["manifest_fingerprint"],
        shape_xyz=list(nib.load(case.image).shape),
        sha256={"ct": file_hash(case.image), "gt": file_hash(case.mask)})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config, spec


def test_actual_trainer_steps_checkpoint_resume_and_ct_only_prediction(tmp_path):
    path, config, spec = fixture_case(tmp_path)
    result = run_trial(path, config["data"]["manifest"], spec, tmp_path / "report.json",
                       tmp_path / "scratch", steps=2, device="cpu", expected_cases=10)
    assert result["status"] == "PASS", result
    assert result["trainer_class"] == "merit_covid.engine.training.Trainer"
    assert result["train"]["optimizer_steps"] == 2 and result["after_resume"]["optimizer_steps"] == 1
    assert result["activation_checkpointing_enabled"] and result["source_files_unchanged"]
    assert result["checkpoint"]["model_exact_match"] and result["checkpoint"]["optimizer_step_counters_match"]
    assert result["inference"]["ct_only_guard"] == "PASS"
    assert not result["validation_evaluated"] and not result["full_training"]
    assert not list((tmp_path / "scratch").rglob("*.pt"))
    assert not result["xy_interpolation_called"] and result["input_matches_native"]


def test_main_trial_rejects_missing_cohort_sources_and_writes_failure(tmp_path):
    path, config, spec = fixture_case(tmp_path)
    records = json.loads(Path(config["data"]["manifest"]).read_text())
    other = next(r for r in records if r["id"] != spec["case_id"])
    (Path(config["data"]["manifest"]).parent / other["mask"]).unlink()
    result = run_trial(path, config["data"]["manifest"], spec, tmp_path / "report.json",
                       tmp_path / "scratch", device="cpu", expected_cases=10)
    assert result["status"] == "FAIL" and "missing mask" in result["error"]
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "FAIL"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_main_trial_actual_cuda_amp_resume(tmp_path):
    path, config, spec = fixture_case(tmp_path, device="cuda")
    result = run_trial(path, config["data"]["manifest"], spec, tmp_path / "report.json",
                       tmp_path / "scratch", steps=2, device="cuda", expected_cases=10)
    assert result["status"] == "PASS", result
    assert result["after_resume"]["optimizer_steps"] == 1
