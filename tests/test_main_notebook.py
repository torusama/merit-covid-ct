"""Main-integrated notebook cells, cache/error handling and small report export."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from tests.test_main_trial import fixture_case


ROOT = Path(__file__).resolve().parents[1]


def cells():
    book = json.loads((ROOT / "notebooks/colab_main_integrated.ipynb").read_text())
    return ["".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code"]


def test_main_notebook_uses_integrated_commit_and_exact_orchestration_helper():
    code = cells()
    namespace = {}
    exec(code[0], namespace)
    assert namespace["SOURCE_COMMIT"] == "ac461691450b2046f19b6c61ba7fb7426cb14884"
    helper = next(c for c in code if "MAIN_TRIAL_SOURCE = " in c)
    tree = ast.parse(helper)
    embedded = ast.literal_eval(tree.body[0].value)
    assert embedded == (ROOT / "tools/merit_main_trial.py").read_text()
    compile(embedded, "merit_main_trial.py", "exec")
    assert "merit_checkpoint_trial" not in embedded


def test_main_notebook_launches_actual_worker_and_exports_no_images_or_weights(tmp_path):
    path, config, spec = fixture_case(tmp_path)
    run = tmp_path / "reports"
    run.mkdir()
    ns = dict(RUN=run, BASE=tmp_path, PROJECT=ROOT, ENVIRONMENT={},
        CASE_SPEC=spec, EXPECTED_CASES=10, TRAIN_STEPS=2, RUN_DEVICE="cpu",
        TRIAL_CONFIG=path, MANIFEST=Path(config["data"]["manifest"]),
        PREPARATION_ERROR=None, Path=Path, json=json, hashlib=hashlib,
        os=os, subprocess=subprocess, sys=sys)
    code = cells()
    exec(next(c for c in code if "MAIN_TRIAL_SOURCE = " in c), ns)
    exec(next(c for c in code if "report_path = RUN" in c), ns)
    assert ns["summary"]["status"] == "PASS", ns["report"]
    assert ns["report"]["after_resume"]["optimizer_steps"] == 1
    assert ns["report"]["production_source_files_unchanged"]
    (run / "must_not_export.pt").write_bytes(b"synthetic model bytes")
    (run / "must_not_export.nii.gz").write_bytes(b"synthetic source bytes")
    exec(next(c for c in code if "import zipfile" in c), ns)
    with zipfile.ZipFile(ns["archive"]) as archive:
        assert "production_main.json" in archive.namelist()
        assert "summary.json" in archive.namelist()
        assert not any(n.endswith((".pt", ".nii.gz")) for n in archive.namelist())


def test_failed_data_preparation_still_generates_failure_report_and_zip(tmp_path):
    run = tmp_path / "reports"
    run.mkdir()
    ns = dict(RUN=run, PROJECT=ROOT, PREPARATION_ERROR="Download interrupted",
        ENVIRONMENT={}, Path=Path, hashlib=hashlib, json=json)
    code = cells()
    exec(next(c for c in code if "report_path = RUN" in c), ns)
    assert ns["summary"]["status"] == "FAIL"
    exec(next(c for c in code if "import zipfile" in c), ns)
    assert ns["archive"].is_file()


def test_complete_cache_indexes_real_files_without_download(tmp_path, monkeypatch):
    import huggingface_hub
    dataset = tmp_path / "dataset"
    (dataset / "Train").mkdir(parents=True)
    data = {"ct": b"synthetic CT bytes", "gt": b"synthetic GT bytes"}
    names = {"ct": "sample_ct.nii.gz", "gt": "sample_seg.nii.gz"}
    for k, content in data.items():
        (dataset / "Train" / names[k]).write_bytes(content)
    (dataset / ".merit_dataset_revision").write_text("fixture-revision\n")
    monkeypatch.setenv("MERIT_DATA_HOME", str(dataset))
    def forbidden_download(*args, **kwargs):
        raise AssertionError("Complete revision must use cached data")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden_download)
    ns = dict(BASE=tmp_path, RUN=tmp_path, Path=Path, os=os, json=json,
        DATASET_REVISION="fixture-revision", EXPECTED_CASES=1,
        CASE_SPEC=dict(ct_name=names["ct"], gt_name=names["gt"],
            file_sizes={k: len(v) for k, v in data.items()},
            sha256={k: hashlib.sha256(v).hexdigest() for k, v in data.items()}))
    exec(next(c for c in cells() if "from huggingface_hub import snapshot_download" in c), ns)
    assert ns["PREPARATION_ERROR"] is None
    records = json.loads(ns["MANIFEST"].read_text())
    assert records == [{"id": "sample", "image": "Train/sample_ct.nii.gz", "mask": "Train/sample_seg.nii.gz"}]
