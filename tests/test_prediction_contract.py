"""One output grid for 3D patches and master-slice models; no metric evaluation."""

from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.preprocessing import read_arrays
from merit_covid.data.pipeline.metrics import SliceReconstructor
from merit_covid.data.unetrpp_input import UNETRPPPatchDataset, PatchConfig
from merit_covid.data.unetrpp_input.tiling import extract_patch
from merit_covid.engine.patch_reconstruction import PatchReconstructor
from merit_covid.engine.patch_inference import predict_patch_batch
from merit_covid.engine.prediction import VolumePrediction, prediction_from_slices, binary_logits_to_probabilities


@pytest.fixture
def volume(nifti_case):
    main, _, _ = nifti_case(shape=(7, 9, 5), affine=np.diag([-0.7, -0.8, -5, 1]))
    case = Case(main.case_id, main.image, None, main.case_id)
    image, _, metadata = read_arrays(case, PreprocessConfig(), require_mask=False)
    field = np.linspace(0, 1, image.size, dtype=np.float32).reshape(image.shape)
    return case, field, metadata


def test_master_slice_and_patch_models_produce_identical_contract(volume):
    case, field, metadata = volume
    slices = [(z, field[z]) for z in reversed(range(field.shape[0]))]
    master = prediction_from_slices(case.case_id, slices, metadata,
                                    policy={'model': 'synthetic master model', 'input': {'slab_size': 11}})
    dataset = UNETRPPPatchDataset([case], PatchConfig(patch_size=(4, 6, 4), stride=(2, 3, 2)))
    patches = PatchReconstructor(case.case_id, dataset.plans[0], dataset[0]['metadata'],
                                 policy={'model': 'synthetic patch model'})
    for i in range(len(dataset)):
        patches.add(i, extract_patch(field, dataset.plans[0][i], dataset.config), metadata=dataset[i]['metadata'])
    patch_output = patches.finish()
    np.testing.assert_allclose(master.probability_zyx, patch_output.probability_zyx, atol=1e-7)
    np.testing.assert_allclose(master.to_original_xyz(), patch_output.to_original_xyz(), atol=1e-7)
    # The exact slice iterator interface consumed by the existing evaluator.
    shared = SliceReconstructor(field.shape)
    for index, plane in patch_output.slice_predictions(): shared.add(index, plane)
    np.testing.assert_array_equal(shared.finish(), patch_output.probability_zyx)
    assert master.metadata['prediction_policy']['input'] == {'slab_size': 11}
    assert patch_output.metadata['prediction_policy']['input']['patch_size'] == [4, 6, 4]
    assert master.metadata['original_affine_mm'] == patch_output.metadata['original_affine_mm']


def test_contract_has_float32_contiguous_probabilities_and_copies_metadata(volume):
    case, field, metadata = volume
    output = VolumePrediction(case.case_id, field.astype(np.float64), metadata)
    assert output.probability_zyx.dtype == np.float32 and output.probability_zyx.flags.c_contiguous
    output.metadata['original_shape_xyz'][0] = 123
    assert metadata['original_shape_xyz'][0] == 7


@pytest.mark.parametrize('kind', ['wrong_shape', 'nan', 'negative', 'above_one', 'empty'])
def test_invalid_unified_probabilities_are_rejected(volume, kind):
    case, field, metadata = volume
    bad = field.copy()
    if kind == 'wrong_shape': bad = bad[:, :-1]
    if kind == 'nan': bad[0, 0, 0] = np.nan
    if kind == 'negative': bad[0, 0, 0] = -0.01
    if kind == 'above_one': bad[0, 0, 0] = 1.01
    if kind == 'empty': bad = bad[:0]
    with pytest.raises(ValueError): VolumePrediction(case.case_id, bad, metadata)


@pytest.mark.parametrize('kind', [
    'missing', 'resized', 'ras', 'affine_ras', 'z_resampled', 'invalid_affine',
    'wrong_transform', 'wrong_original_shape', 'wrong_case',
])
def test_metadata_must_describe_native_lps_and_original_geometry(volume, kind):
    case, field, metadata = volume
    bad = deepcopy(metadata)
    if kind == 'missing': del bad['original_affine']
    if kind == 'resized': bad['processed_size_hw'] = [4, 4]
    if kind == 'ras': bad['canonical_orientation'] = ['R', 'A', 'S']
    if kind == 'affine_ras': bad['canonical_affine_mm'][0][0] *= -1
    if kind == 'z_resampled': bad['z_resampled'] = True
    if kind == 'invalid_affine': bad['original_affine_mm'][0][0] = 0
    if kind == 'wrong_transform': bad['canonical_to_original_orientation'][0][1] *= -1
    if kind == 'wrong_original_shape': bad['original_shape_xyz'][0] = 999
    if kind == 'wrong_case': bad['case_id'] = 'other'
    with pytest.raises(ValueError): VolumePrediction(case.case_id, field, bad)


def test_master_slice_bridge_rejects_missing_and_duplicate_slices(volume):
    case, field, metadata = volume
    with pytest.raises(ValueError, match='Missing'):
        prediction_from_slices(case.case_id, [(0, field[0])], metadata)
    with pytest.raises(ValueError, match='Duplicate'):
        prediction_from_slices(case.case_id, [(0, field[0]), (0, field[0])], metadata)


@pytest.mark.parametrize('shape', [(2, 1, 3, 4), (2, 1, 3, 4, 5)])
def test_stable_sigmoid_in_output_layer_only(shape):
    logits = np.linspace(-1000, 1000, np.prod(shape), dtype=np.float32).reshape(shape)
    before = logits.copy()
    probs = binary_logits_to_probabilities(logits)
    assert probs.shape == (shape[0], *shape[2:]) and probs.dtype == np.float32
    assert np.isfinite(probs).all() and probs.min() == 0 and probs.max() == 1
    np.testing.assert_array_equal(logits, before)
    torch_result = binary_logits_to_probabilities(torch.from_numpy(logits).requires_grad_())
    np.testing.assert_array_equal(probs, torch_result)


@pytest.mark.parametrize('logits', [
    np.zeros((1, 2, 3, 4), dtype=np.float32), np.zeros((1, 3, 4), dtype=np.float32),
    np.zeros((1, 1, 3, 4), dtype=np.int16), np.full((1, 1, 3, 4), np.inf),
    np.full((1, 1, 3, 4), np.nan), np.zeros((0, 1, 3, 4), dtype=np.float32),
])
def test_invalid_binary_logits(logits):
    with pytest.raises(ValueError): binary_logits_to_probabilities(logits)


class InvalidOutputModel(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, image):
        if self.kind == 'wrong_shape': return image[:, :, :-1]
        if self.kind == 'nan': return image * float('nan')
        if self.kind == 'inf': return image + float('inf')
        return 'not a tensor'


@pytest.mark.parametrize('kind', ['wrong_shape', 'nan', 'inf', 'not_tensor'])
def test_patch_forward_rejects_invalid_model_output(kind):
    with pytest.raises(ValueError):
        predict_patch_batch(InvalidOutputModel(kind).eval(), {'image_patch': torch.zeros(1, 1, 2, 3, 4)})


def test_new_geometry_and_output_modules_keep_torch_optional():
    source = """
import sys
sys.modules['torch'] = None
sys.modules['monai'] = None
sys.modules['einops'] = None
from merit_covid.data.unetrpp_input import PatchConfig, PatchPlan, UNETRPPPatchDataset
from merit_covid.engine.prediction import VolumePrediction, prediction_from_slices
from merit_covid.engine.patch_reconstruction import PatchReconstructor
assert len(PatchPlan((1,2,3), PatchConfig())) == 1
"""
    subprocess.run([sys.executable, '-c', source], check=True, capture_output=True, text=True)
