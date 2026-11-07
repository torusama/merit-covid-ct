"""Small synthetic fixtures for software tests only; never dataset/experiment results."""
import json

import nibabel as nib
import numpy as np
import pytest

from ct_pipeline.cases import Case


@pytest.fixture
def write_case(tmp_path):
    def write(case_id="fixture", shape=(8, 6, 13), affine=None, ct=None, mask=None, unit="mm"):
        affine = np.diag([-1.0, -2.0, 5.0, 1.0]) if affine is None else affine
        if ct is None:
            x, y, z = np.indices(shape)
            ct = (-1000 + 100 * z + 20 * x + 10 * y).astype(np.float32)
        if mask is None:
            mask = np.zeros(shape, dtype=np.uint8)
            mask[2:4, 2:4, 1] = 1
        paths = [tmp_path / f"{case_id}_{suffix}.nii.gz" for suffix in ("ct", "seg")]
        for path, array in zip(paths, (ct, mask)):
            image = nib.Nifti1Image(array, affine)
            image.header.set_xyzt_units(unit)
            nib.save(image, path)
        return Case(case_id, *paths, case_id)
    return write


@pytest.fixture
def grouped_split(write_case, tmp_path):
    """10 series nhỏ thuộc 5 bệnh nhân giả lập, chỉ phục vụ unit tests."""
    from ct_pipeline.cases import load_cases
    from ct_pipeline.splits import create_splits

    cases = [write_case(f"series_{index}") for index in range(10)]
    records = [
        {
            "id": case.case_id,
            "patient_id": f"patient_{index // 2}",
            "image": case.image.name,
            "mask": case.mask.name,
        }
        for index, case in enumerate(cases)
    ]
    manifest = tmp_path / "cases.json"
    split_path = tmp_path / "split.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    grouped_cases, identity = load_cases(manifest)
    split_path.write_text(
        json.dumps(create_splits(grouped_cases, identity)), encoding="utf-8"
    )
    return manifest, split_path
