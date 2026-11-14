"""Input ZIP extraction/cache safety without a Colab session or patient images."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types
import zipfile

import pytest


@pytest.mark.parametrize("extra_member", [False, True])
def test_upload_zip_is_validated_and_correct_inputs_are_cached(tmp_path, monkeypatch, extra_member):
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks/colab_real_case.ipynb").read_text())
    upload_code = next("".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "def inputs_ok(" in "".join(c["source"]))
    data = {"ct.nii.gz": b"synthetic ct bytes", "gt.nii.gz": b"synthetic gt bytes"}
    archive = tmp_path / "upload.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name, content in data.items():
            handle.writestr(name, content)
        if extra_member:
            handle.writestr("../escape.txt", "must not be extracted")
    calls = []
    def upload():
        calls.append(True)
        shutil.copyfile(archive, Path.cwd() / archive.name)
        return {archive.name: b""}
    google, colab = types.ModuleType("google"), types.ModuleType("google.colab")
    google.__path__ = []
    colab.files = types.SimpleNamespace(upload=upload)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    import os
    spec = {"ct_name": "ct.nii.gz", "gt_name": "gt.nii.gz",
        "sha256": {k: hashlib.sha256(data[n]).hexdigest() for k, n in (("ct", "ct.nii.gz"), ("gt", "gt.nii.gz"))},
        "file_sizes": {"ct": len(data["ct.nii.gz"]), "gt": len(data["gt.nii.gz"])}}
    namespace = {"BASE": tmp_path, "CASE_SPEC": spec, "Path": Path, "os": os,
                 "hashlib": hashlib, "json": json}
    before_cwd = Path.cwd()
    exec(upload_code, namespace)
    assert Path.cwd() == before_cwd and not (tmp_path / "escape.txt").exists()
    if extra_member:
        assert "exactly the two expected" in namespace["UPLOAD_ERROR"]
    else:
        assert namespace["UPLOAD_ERROR"] is None
        assert namespace["CT_PATH"].read_bytes() == data["ct.nii.gz"]
        exec(upload_code, namespace)
        assert len(calls) == 1
