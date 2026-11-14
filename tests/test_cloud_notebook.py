"""Portable notebook bootstrap contracts; no internet, GPU or pip mutations."""

import ast
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from packaging.specifiers import SpecifierSet
import pytest


ROOT = Path(__file__).resolve().parents[1]


def code_cells():
    notebook = json.loads((ROOT / "notebooks/colab_connect.ipynb").read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.mark.parametrize("filename", ["colab_connect.ipynb", "colab_checkpoint_trial.ipynb", "colab_real_case.ipynb", "colab_mer_only_case.ipynb", "colab_main_integrated.ipynb"])
def test_notebook_and_generated_worker_compile_without_saved_outputs(filename):
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["metadata"]["accelerator"] == "GPU"
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [] and cell["execution_count"] is None
        source = "".join(cell["source"])
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "worker"
                    and node.func.attr == "write_text"):
                compile(ast.literal_eval(node.args[0]), "profile_worker.py", "exec")


@pytest.mark.parametrize("filename", ["colab_checkpoint_trial.ipynb", "colab_real_case.ipynb", "colab_mer_only_case.ipynb"])
def test_checkpoint_notebook_embeds_the_validated_helper_exactly(filename):
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    found = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TRIAL_SOURCE" for t in node.targets):
                found.append(ast.literal_eval(node.value))
    assert found == [(ROOT / "tools/merit_checkpoint_trial.py").read_text(encoding="utf-8")]


@pytest.mark.parametrize("filename", ["colab_real_case.ipynb", "colab_mer_only_case.ipynb"])
def test_real_notebook_embeds_the_validated_pilot_exactly(filename):
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    found = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REAL_TRIAL_SOURCE" for t in node.targets):
                found.append(ast.literal_eval(node.value))
    assert found == [(ROOT / "tools/merit_real_case_trial.py").read_text(encoding="utf-8")]


def test_mer_only_notebook_embeds_candidate_and_selects_its_worker():
    notebook = json.loads((ROOT / "notebooks/colab_mer_only_case.ipynb").read_text(encoding="utf-8"))
    found, worker = [], []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "command = [" in source:
            worker.append(source)
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CANDIDATE_SOURCE" for t in node.targets):
                found.append(ast.literal_eval(node.value))
    assert found == [(ROOT / "tools/merit_mer_only_trial.py").read_text(encoding="utf-8")]
    compile(found[0], "merit_mer_only_trial.py", "exec")
    assert len(worker) == 1 and "str(CANDIDATE_FILE)" in worker[0]


def test_gpu_not_assigned_stops_before_download_or_install(monkeypatch):
    calls = []
    def cpu_probe(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({
            "torch": "2.7.1+cpu", "cuda_available": False, "gpus": [], "cuda": None}))
    monkeypatch.setattr(subprocess, "run", cpu_probe)
    with pytest.raises(RuntimeError, match="Enable a GPU"):
        exec(code_cells()[1], {})
    assert len(calls) == 1 and calls[0][1] == "-c"


def test_cached_bootstrap_installs_only_missing_packages_and_protects_runtime(tmp_path, monkeypatch):
    cells = code_cells()
    namespace = {}
    exec(cells[0], namespace)
    source = tmp_path / ("merit-source-" + namespace["SOURCE_COMMIT"][:12])
    (source / ".git").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "scripts/merit_preflight.py").write_text("# synthetic checkout marker\n")
    monkeypatch.setenv("MERIT_CLOUD_HOME", str(tmp_path))
    installed = {"torch": "2.7.1+cu118", "numpy": "1.26.4", "einops": "0.8.2",
                 "nibabel": "5.3.0", "scipy": "1.14.1", "PyYAML": "6.0.2"}
    def version(name):
        if name not in installed:
            raise metadata.PackageNotFoundError(name)
        return installed[name]
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[:3] == [sys.executable, "-m", "pip"]:
            assert command[-1] == "monai==1.5.2"
            constraints = Path(command[command.index("--constraint") + 1]).read_text()
            assert constraints == "torch==2.7.1+cu118\nnumpy==1.26.4\n"
            installed["monai"] = "1.5.2"
        elif command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout=namespace["SOURCE_COMMIT"] + "\n")
        else:
            assert command[0] == "git" and "clone" not in command and "fetch" not in command
        return subprocess.CompletedProcess(command, 0, stdout="")
    monkeypatch.setattr(subprocess, "run", run)
    namespace.update(json=json, subprocess=subprocess, sys=sys, SpecifierSet=SpecifierSet,
                     metadata=SimpleNamespace(version=version, PackageNotFoundError=metadata.PackageNotFoundError),
                     ENVIRONMENT={"cuda_available": True})
    exec(cells[2], namespace)
    assert len([c for c in calls if c[:3] == [sys.executable, "-m", "pip"]]) == 1
    calls.clear()
    exec(cells[2], namespace)
    assert not any(c[:3] == [sys.executable, "-m", "pip"] for c in calls)
