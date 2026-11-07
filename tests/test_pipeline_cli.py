"""Kiểm tra CLI sau tách run_*; prediction tổng hợp chỉ dùng cho tests."""

import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np

from ct_pipeline.cases import load_cases
from ct_pipeline.geometry import reorient_to_lps


CLI = Path(__file__).resolve().parents[1] / "tools/data_pipeline.py"


def run_cli(*arguments):
    return subprocess.run(
        [sys.executable, str(CLI), *map(str, arguments)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_audit_cli_keeps_unknown_units_strict_and_override_explicit(write_case, tmp_path):
    case = write_case(unit="unknown")
    manifest = tmp_path / "cases.json"
    manifest.write_text(
        json.dumps([{"id": case.case_id, "image": case.image.name, "mask": case.mask.name}]),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    arguments = [
        "audit", "--manifest", manifest, "--expected-cases", 1,
        "--output", output,
    ]
    strict = run_cli(*arguments)
    assert strict.returncode == 1
    strict_report = json.loads(output.read_text(encoding="utf-8"))
    assert strict_report["valid_pairs"] == 0
    assert strict_report["invalid_pairs"] == 1
    assert "Spatial unit" in strict_report["failures"][0]["error"]

    overridden = run_cli(*arguments, "--assume-mm")
    assert overridden.returncode == 0, overridden.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid_pairs"] == 1 and report["invalid_pairs"] == 0
    assert report["cases_using_assumed_mm"] == [case.case_id]
    assert report["per_case"][0]["label_values"] == [0, 1]
    assert report["config"]["size_hw"] is None
    assert report["per_case"][0]["canonical_orientation"] == ["L", "P", "S"]
    assert report["per_case"][0]["processed_size_hw"] == [6, 8]


def test_evaluate_cli_reads_only_selected_partition_with_unordered_slices(grouped_split, tmp_path):
    manifest, split_path = grouped_split
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = set(split["folds"][0]["test"]["case_ids"])
    cases, _ = load_cases(manifest)
    predictions = tmp_path / "predictions"
    predictions.mkdir()

    # Hai test series nhận GT giả lập hoàn hảo, thứ tự slice đảo ngược.
    # Các ca khác cố ý có archive hỏng: CLI không được đọc chúng.
    for case in cases:
        path = predictions / f"{case.case_id}.npz"
        if case.case_id in test_ids:
            canonical = reorient_to_lps(nib.load(case.mask))
            labels = np.asarray(canonical.dataobj).transpose(2, 1, 0)
            np.savez(
                path,
                indices=np.arange(len(labels))[::-1],
                probabilities=labels[::-1],
            )
        else:
            path.write_bytes(b"Not a valid NPZ; must be ignored outside test partition")

    output = tmp_path / "metrics.json"
    result = run_cli(
        "evaluate", "--manifest", manifest, "--expected-cases", 10,
        "--split", split_path, "--fold", 0, "--partition", "test",
        "--predictions", predictions, "--output", output, "--hd95",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert {row["case_id"] for row in report["per_case"]} == test_ids
    assert report["macro"]["dice"]["mean"] == 1
    assert report["macro"]["hd95_mm"]["mean"] == 0
    assert report["macro"]["dice"]["valid_groups"] == 1
    assert report["fold"] == 0 and report["partition"] == "test"
