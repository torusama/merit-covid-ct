"""Small synthetic fixtures for software tests only; never dataset/experiment results."""
import nibabel as nib
import numpy as np
import pytest

from ct_pipeline.cases import Case


@pytest.fixture
def write_case(tmp_path):
    def write(case_id="fixture", shape=(8, 6, 13), affine=None, ct=None, mask=None, unit="mm"):
        affine = np.diag([1.0, 2.0, 5.0, 1.0]) if affine is None else affine
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
