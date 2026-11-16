"""Bounded pilot behavior and portable notebook source/failure reporting."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import yaml

from tools.merit_attention_pilot import run_pilot
from tests.test_real_case_trial import pilot

ROOT = Path(__file__).resolve().parents[1]


def code_cells():
    notebook = json.loads((ROOT / "notebooks/colab_mer_attention_pilot.ipynb").read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_portable_notebook_embeds_current_source_and_only_three_data_identities():
    cells = code_cells()
    for index, source in enumerate(cells):
        compile(source, f"cell_{index}", "exec")
    tree = ast.parse(cells[0])
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body[:6]}
    raw = base64.b64decode(values["SOURCE_BUNDLE_B64"])
    assert hashlib.sha256(raw).hexdigest() == values["SOURCE_SHA256"]
    assert len(values["CASE_SPECS"]) == 3
    assert [s["case_id"][-4:] for s in values["CASE_SPECS"]] == ["0025", "0295", "0387"]
    assert sum(sum(s["file_sizes"].values()) for s in values["CASE_SPECS"]) == 47947366
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert not any(name.endswith((".nii", ".nii.gz", ".pt")) for name in archive.namelist())
        for name in ("tools/merit_attention_pilot.py", "src/merit_covid/models/merit_unetrpp/modules.py",
                     "src/merit_covid/engine/activation_checkpointing.py", "tests/test_mer_attention_execution.py"):
            assert archive.read(name) == (ROOT / name).read_bytes()


def test_bounded_cpu_pilot_compares_same_init_and_checks_checkpoint(pilot, tmp_path):
    path, _, spec, case, _ = pilot
    config = yaml.safe_load(path.read_text())
    config["preprocess"]["size_hw"] = None
    config["training"]["activation_checkpointing"] = True
    path.write_text(yaml.safe_dump(config))
    spec = dict(spec, ct_name=case.image.name, gt_name=case.mask.name, masters=[6, 0])
    reports = [run_pilot(path, case.image.parent, [spec], tmp_path / (mode + ".json"),
                         variant=mode, device="cpu", rounds=1) for mode in ("legacy", "optimized")]
    for report in reports:
        assert report["status"] == "PASS", report
        assert report["source_files_unchanged"] and report["weights_changed"]
        assert report["measured_steps"] == 2 and len(report["batches"]) == 3
        assert [r["master_positive"] for r in report["input_checks"]] == [True, False]
        assert all(r["gradients_finite"] and not r["amp_skipped"] for r in report["batches"])
    assert reports[0]["initial_state_sha256"] == reports[1]["initial_state_sha256"]
    assert reports[0]["final_state_sha256"] == reports[1]["final_state_sha256"]
    assert reports[1]["diagnostic_checkpoint"]["model_exact"]
    assert reports[1]["diagnostic_checkpoint"]["optimizer_steps_exact"]
    assert reports[1]["diagnostic_checkpoint_deleted"]
    assert reports[1]["inference"]["ct_only_guard"] == "PASS"


def test_notebook_failure_still_exports_small_report_zip(tmp_path):
    (tmp_path / "must_not_export.pt").write_bytes(b"weights")
    (tmp_path / "must_not_export.nii.gz").write_bytes(b"image")
    (tmp_path / "worker.log").write_text("fixture failure")
    namespace = dict(RUN=tmp_path, PROJECT=ROOT, SOURCE_FILES={}, REPORTS={}, PREPARATION_ERROR="fixture download error",
        REGRESSION=dict(status="NOT_RUN"), SOURCE_SHA256="fixture", CASE_SPECS=[], MEASURED_ROUNDS=2,
        hashlib=hashlib, json=json, zipfile=zipfile)
    exec(next(c for c in code_cells() if "# Luôn tạo ZIP báo cáo" in c), namespace)
    assert namespace["summary"]["status"] == "FAIL"
    with zipfile.ZipFile(namespace["archive_path"]) as archive:
        assert set(archive.namelist()) == {"summary.json", "worker.log"}


def test_runtime_profile_cell_embeds_current_diagnostic_without_repeating_pilot():
    source = next(c for c in code_cells() if c.startswith("PROFILE_SOURCE = "))
    tree = ast.parse(source)
    embedded = ast.literal_eval(tree.body[0].value)
    expected_sha = ast.literal_eval(tree.body[1].value)
    assert embedded == (ROOT / "tools/merit_runtime_profile.py").read_text(encoding="utf-8")
    assert hashlib.sha256(embedded.encode()).hexdigest() == expected_sha
    compile(embedded, "merit_runtime_profile.py", "exec")
    assert 'for mode in ("auto", "efficient")' in source
    assert '"tools/merit_attention_pilot.py"' not in source


def test_factorized_cell_embeds_current_candidate_and_preserves_source():
    source = next(c for c in code_cells() if c.startswith("FACTOR_SOURCES = "))
    tree = ast.parse(source)
    embedded = ast.literal_eval(tree.body[0].value)
    hashes = ast.literal_eval(tree.body[1].value)
    for name, data in embedded.items():
        assert data == (ROOT / name).read_text(encoding="utf-8")
        assert hashlib.sha256(data.encode()).hexdigest() == hashes[name]
        compile(data, name, "exec")
    assert "shutil.copytree(PROJECT, CANDIDATE_PROJECT" in source


def test_cpu_runtime_profile_captures_attention_and_backward(pilot, tmp_path):
    from tools.merit_runtime_profile import run_profile
    path, _, spec, case, _ = pilot
    config = yaml.safe_load(path.read_text())
    config["preprocess"]["size_hw"] = None
    config["training"]["activation_checkpointing"] = True
    path.write_text(yaml.safe_dump(config))
    spec = dict(spec, ct_name=case.image.name, gt_name=case.mask.name, masters=[6, 0])
    report = run_profile(path, case.image.parent, [spec], tmp_path / "profile.json", device="cpu")
    assert report["status"] == "PASS", report
    assert len(report["batches"]) == 4 and report["source_files_unchanged"]
    assert all(b["gradients_finite"] and not b["amp_skipped"] for b in report["batches"])
    assert report["attention_ops"] and report["top_self_cpu"]
    assert any(e["name"] == "merit_phase::backward" for e in report["stages"])
    assert report["mer_first_call"]["key_shape"][1] > 0


def test_factorized_candidate_runs_production_model_and_checkpoint(pilot, tmp_path):
    from tools.merit_factorized_pilot import run_factorized
    path, _, spec, case, _ = pilot
    config = yaml.safe_load(path.read_text())
    config["preprocess"]["size_hw"] = None
    config["training"]["activation_checkpointing"] = True
    path.write_text(yaml.safe_dump(config))
    spec = dict(spec, ct_name=case.image.name, gt_name=case.mask.name, masters=[6, 0])
    report = run_factorized(path, case.image.parent, [spec], tmp_path / "factorized.json", device="cpu")
    assert report["status"] == "PASS", report
    assert report["source_files_unchanged"] and report["weights_changed"]
    assert report["diagnostic_checkpoint"]["model_exact"]
    assert report["diagnostic_checkpoint"]["optimizer_steps_exact"]
    assert report["inference"]["ct_only_guard"] == "PASS"
