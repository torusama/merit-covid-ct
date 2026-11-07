import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np

from merit_covid.data.manifest import load_manifest, write_manifest
from merit_covid.data.splitting import load_splits


ROOT = Path(__file__).resolve().parents[1]


def run_script(name, *arguments):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
                          cwd=ROOT, capture_output=True, text=True)


def test_checker_full_and_enriched_manifest(nifti_case, tmp_path):
    case, _, _ = nifti_case()
    manifest = tmp_path / "manifest.json"
    write_manifest([case], manifest)
    enriched = tmp_path / "enriched.json"
    result = run_script("check_dataset.py", manifest, "--write-manifest", enriched)
    assert result.returncode == 0, result.stderr
    assert "1/1" in result.stdout
    output = load_manifest(enriched)[0]
    assert output.shape == (6, 8, 4)
    assert output.affine is not None
    assert output.patient_id is None
    assert output.intensity == "unknown"
    again = run_script("check_dataset.py", manifest, "--write-manifest", enriched)
    assert again.returncode != 0
    assert "FileExistsError" in again.stderr


def test_header_check_does_not_claim_voxel_validation(nifti_case, tmp_path):
    case, _, mask = nifti_case()
    image = nib.load(case.infection_mask)
    invalid = mask.copy()
    invalid[0, 0, 0] = 3
    nib.save(nib.Nifti1Image(invalid, image.affine, image.header), case.infection_mask)
    manifest = tmp_path / "manifest.json"
    write_manifest([case], manifest)
    headers = run_script("check_dataset.py", manifest, "--headers-only")
    assert headers.returncode == 0, headers.stderr
    assert "voxel labels NOT checked" in headers.stdout
    full = run_script("check_dataset.py", manifest)
    assert full.returncode != 0
    assert "Expected binary infection mask" in full.stderr


def test_split_cli_with_only_synthetic_cases(nifti_case, tmp_path):
    cases = [nifti_case(case_id=f"v{i}")[0] for i in range(10)]
    manifest = tmp_path / "manifest.json"
    write_manifest(cases, manifest)
    target = tmp_path / "folds.json"
    result = run_script("make_splits.py", manifest, "--expected-count", 10, "--output", target)
    assert result.returncode == 0, result.stderr
    assert len(load_splits(target, cases)["folds"]) == 5
    again = run_script("make_splits.py", manifest, "--expected-count", 10, "--output", target)
    assert again.returncode != 0
    assert "FileExistsError" in again.stderr
