from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from merit_covid.data.manifest import Case


@pytest.fixture
def nifti_case(tmp_path):
    def create(*, shape=(6, 8, 4), affine=None, labels=None, case_id="case1", dataset="covid19_20", unit="mm"):
        folder = tmp_path / case_id
        folder.mkdir(exist_ok=True)
        if affine is None:
            affine = np.diag([-0.7, 0.8, -5.0, 1.0])
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        mask = np.zeros(shape, dtype=np.uint8) if labels is None else np.array(labels)
        if labels is None:
            mask[1, 2, 1] = 1
        for name, values in (("ct", image), ("lesion", mask)):
            result = nib.Nifti1Image(values, affine)
            result.header.set_xyzt_units(unit)
            nib.save(result, folder / f"{name}.nii.gz")
        return Case(dataset, case_id, folder / "ct.nii.gz", folder / "lesion.nii.gz"), image, mask
    return create
