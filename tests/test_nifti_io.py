from dataclasses import replace

import nibabel as nib
import numpy as np
import pytest

from merit_covid.data.nifti_io import binary_mask, load_case, save_prediction
from merit_covid.data.preprocessing import preprocess, restore_prediction
from merit_covid.config import Config


def test_geometry_order_and_original_roundtrip(nifti_case, tmp_path):
    case, image, mask = nifti_case()
    volume = load_case(case)
    assert volume.original_orientation == ("L", "A", "I")
    assert nib.aff2axcodes(volume.affine) == ("L", "P", "S")
    np.testing.assert_allclose(volume.spacing_mm, [0.7, 0.8, 5])
    np.testing.assert_array_equal(volume.image, image[:, ::-1, ::-1])
    prepared = preprocess(volume, Config())
    restored = restore_prediction(prepared.mask, prepared)
    np.testing.assert_array_equal(restored, mask)
    target = tmp_path / "prediction.nii.gz"
    save_prediction(restored, volume, target)
    saved = nib.load(target)
    np.testing.assert_allclose(saved.affine, volume.original_affine)
    np.testing.assert_array_equal(np.asanyarray(saved.dataobj), mask)
    assert saved.header.get_xyzt_units()[0] == "mm"
    assert saved.header["sform_code"] == volume.original_header["sform_code"]
    with pytest.raises(FileExistsError):
        save_prediction(restored, volume, target)


@pytest.mark.parametrize("kind", ["shape", "affine", "units"])
def test_misalignment_rejected(nifti_case, kind):
    case, _, mask = nifti_case()
    affine = np.diag([-0.7, 0.8, -5.0, 1.0])
    if kind == "shape":
        mask = mask[:, :, :2]
    if kind == "affine":
        affine[0, 3] += 2
    result = nib.Nifti1Image(mask, affine)
    result.header.set_xyzt_units("meter" if kind == "units" else "mm")
    nib.save(result, case.infection_mask)
    with pytest.raises(ValueError, match="mismatch"):
        load_case(case)


def test_unknown_units_require_evidence(nifti_case):
    case, _, _ = nifti_case(unit="unknown")
    with pytest.raises(ValueError, match="unknown physical units"):
        load_case(case)
    np.testing.assert_allclose(load_case(case, assumed_unit="mm").spacing_mm, [0.7, 0.8, 5])


def test_meter_geometry(nifti_case):
    case, _, _ = nifti_case(unit="meter", affine=np.diag([0.0007, 0.0008, 0.005, 1]))
    np.testing.assert_allclose(load_case(case).spacing_mm, [0.7, 0.8, 5], rtol=1e-6)


def test_strict_labels_and_explicit_combined_mapping():
    values = np.array([0, 1, 2, 3])
    with pytest.raises(ValueError, match="binary infection"):
        binary_mask(values)
    np.testing.assert_array_equal(binary_mask(values, (3,)), [0, 0, 0, 1])
    for values in (np.array([0, 0.5]), np.array([0, np.nan])):
        with pytest.raises(ValueError):
            binary_mask(values)


def test_stale_manifest_and_nonfinite(nifti_case):
    case, image, _ = nifti_case()
    with pytest.raises(ValueError, match="stale manifest shape"):
        load_case(replace(case, shape=(1, 2, 3)))
    image[0, 0, 0] = np.nan
    original = nib.load(case.image)
    nib.save(nib.Nifti1Image(image, original.affine, original.header), case.image)
    with pytest.raises(ValueError, match="NaN or Inf"):
        load_case(case)
