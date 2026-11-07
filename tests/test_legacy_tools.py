"""Exercise legacy workflows only with temporary synthetic files or mocked I/O."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import nibabel as nib
import numpy as np
import pytest


def import_tool(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_downloader_dispatch_no_network(monkeypatch):
    module = import_tool("download_datasets")
    calls = []
    for name in ("download_covid19_20", "download_zenodo20", "prepare_zenodo20"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(main=lambda name=name: calls.append(name)))
    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "all"])
    module.main()
    assert calls == ["download_zenodo20", "prepare_zenodo20", "download_covid19_20"]


def test_prepare_preserves_actual_names_and_blocks_archive_paths(tmp_path):
    module = import_tool("prepare_zenodo20")
    archive = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../elsewhere/coronacases_org_001.nii", b"synthetic fixture, not loaded")
        handle.writestr("unrelated.txt", b"ignore")
    destination = tmp_path / "extracted"
    module.extract_nifti_files(archive, destination)
    assert list(destination.iterdir()) == [destination / "coronacases_org_001.nii"]
    assert module.indexed_nifti_files(destination) == {"coronacases_001": destination / "coronacases_org_001.nii"}
    assert module.case_id("radiopaedia_org_example-10_85902_1-dcm.nii.gz") == "radiopaedia_10_85902_1"
    assert module.case_id("case009.nii") == "coronacases_009"


def test_audit_consumes_explicit_mask_paths(nifti_case, tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    module = import_tool("audit_zenodo20")
    rows = []
    for number, source in enumerate(("Coronacases", "Radiopaedia")):
        case, _, mask = nifti_case(case_id=f"synthetic{number}", dataset="zenodo20")
        image = nib.load(case.image)
        lung_path = case.image.with_name("actual_lung_source.nii")
        combined_path = case.image.with_name("actual_combined_source.nii")
        combined = np.ones(mask.shape, dtype=np.uint8)
        combined[mask == 1] = 3
        for path, values in ((lung_path, np.ones_like(mask)), (combined_path, combined)):
            nib.save(nib.Nifti1Image(values, image.affine, image.header), path)
        rows.append(dict(id=case.case_id, source=source, image=str(case.image), mask=str(case.infection_mask),
                         lung_mask=str(lung_path), combined_mask=str(combined_path)))
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps(rows))
    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path / "audit")
    module.main()
    summary = json.loads((tmp_path / "audit" / "summary.json").read_text())
    assert summary["cases"] == 2
    assert summary["geometry_match_all"]
    assert summary["infection_encoding_match_all"]


def test_viewer_external_panels_with_recorded_paths(nifti_case, tmp_path):
    pytest.importorskip("tkinter")
    pytest.importorskip("PIL")
    module = import_tool("view_slices")
    case, _, _ = nifti_case(dataset="zenodo20")
    row = dict(id=case.case_id, image=str(case.image), mask=str(case.infection_mask),
               lung_mask=str(case.infection_mask), combined_mask=str(case.infection_mask), shape=[6, 8, 4])
    spec = module.DatasetSpec("synthetic", "Synthetic external", tmp_path, True)
    loaded = module.load_case_images(spec, row)
    assert len(module.render_panels(spec, loaded, 1)) == 5
