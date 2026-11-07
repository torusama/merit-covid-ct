from dataclasses import replace

import numpy as np
import pytest

from merit_covid.config import Config
from merit_covid.data.nifti_io import load_case
from merit_covid.data.preprocessing import normalize, preprocess, resize_xy, restore_prediction


def test_intensity_provenance():
    image = np.array([-1100, -1000, -300, 400, 600], dtype=np.float32)
    with pytest.raises(ValueError, match="verified"):
        normalize(image, "hu_window", intensity="unknown")
    np.testing.assert_allclose(normalize(image, "hu_window", intensity="HU", hu_window=(-1000,400)), [0, 0, 0.5, 1, 1])
    np.testing.assert_array_equal(normalize(image, "none", intensity="unknown"), image)
    with pytest.raises(ValueError):
        normalize(image, "hu_window", intensity="HU")


def test_resize_binary_and_no_z_resampling(nifti_case):
    case, _, _ = nifti_case()
    volume = load_case(case)
    prepared = preprocess(volume, Config(image_size=(12, 16)))
    assert prepared.image.shape == (16, 12, 4)
    assert set(np.unique(prepared.mask)) == {0, 1}
    assert prepared.mask.sum() == volume.mask.sum() * 4
    np.testing.assert_allclose(prepared.affine[:3, 2], volume.affine[:3, 2])
    np.testing.assert_array_equal(restore_prediction(prepared.mask, prepared), volume.mask[:, ::-1, ::-1])


def test_small_lesion_loss_is_visible(nifti_case):
    case, _, _ = nifti_case()
    volume = load_case(case)
    with pytest.warns(UserWarning, match="removed all lesion"):
        prepared = preprocess(volume, Config(image_size=(1, 1)))
    assert prepared.lost_positive_slices
    assert set(np.unique(prepared.mask)) <= {0, 1}


def test_permuted_orientation_inverse(nifti_case):
    affine = np.array([[0, 0, -0.7, 4], [0.8, 0, 0, 1], [0, -5, 0, 30], [0, 0, 0, 1]], dtype=float)
    case, _, mask = nifti_case(affine=affine)
    prepared = preprocess(load_case(case), Config())
    assert prepared.image.shape == (4, 6, 8)
    np.testing.assert_array_equal(restore_prediction(prepared.mask, prepared), mask)


def test_probability_inverse(nifti_case):
    case, _, _ = nifti_case()
    prepared = preprocess(load_case(case), Config(image_size=(12, 16)))
    prediction = np.full(prepared.image.shape, 0.25, np.float32)
    restored = restore_prediction(prediction, prepared, binary=False)
    assert restored.shape == (6, 8, 4)
    np.testing.assert_allclose(restored, 0.25)
