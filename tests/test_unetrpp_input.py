"""Synthetic geometry/assembly tests. Real CT forward is a separate smoke example."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.preprocessing import read_arrays
from merit_covid.data.pipeline.splits import create_splits
from merit_covid.data.unetrpp_input import PatchConfig, PatchPlan, UNETRPPPatchDataset, make_patch_dataloader
from merit_covid.data.unetrpp_input.dataset import load_patch_cases
from merit_covid.data.unetrpp_input.tiling import axis_starts, extract_patch, valid_region_mask
from merit_covid.engine.patch_reconstruction import PatchReconstructor
from merit_covid.engine.patch_inference import predict_patch_batch, predict_patch_case
from merit_covid.engine.prediction import VolumePrediction, binary_logits_to_probabilities


def source_case(case):
    return Case(case.case_id, case.image, case.infection_mask, case.case_id)


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_configuration_and_overlap_policy():
    assert PatchConfig().effective_stride == (64, 128, 128)
    config = PatchConfig(patch_size=[8, 12, 16], overlap=[0.5, 0.25, 0.0])
    assert config.patch_size == (8, 12, 16) and config.effective_stride == (4, 9, 16)
    assert PatchConfig(patch_size=(2, 2, 2), overlap=(0.999, 0.999, 0.999)).effective_stride == (1, 1, 1)
    policy = config.to_dict()
    assert policy['resize'] is False and policy['normalization'] is None and policy['z_resampling'] is False
    json.dumps(policy, allow_nan=False)


@pytest.mark.parametrize('kwargs', [
    {'patch_size': (0, 2, 3)}, {'patch_size': (2, 3)}, {'patch_size': (True, 2, 3)},
    {'stride': (65, 128, 128)}, {'stride': (0, 2, 3)}, {'stride': (1, 2)},
    {'overlap': (1.0, 0, 0)}, {'overlap': (-0.1, 0, 0)}, {'overlap': (float('nan'), 0, 0)},
    {'overlap': (0.5, 0.5)}, {'stride': (1, 2, 3), 'overlap': (0, 0, 0)},
    {'padding_mode': 'reflect'}, {'padding_value': float('inf')}, {'assume_mm': 1},
])
def test_invalid_patch_policy(kwargs):
    with pytest.raises(ValueError):
        PatchConfig(**kwargs)


@pytest.mark.parametrize('length,size,stride,expected', [
    (61, 64, 64, (0,)), (64, 64, 64, (0,)), (65, 64, 64, (0, 1)),
    (130, 64, 64, (0, 64, 66)), (130, 64, 32, (0, 32, 64, 66)),
    (512, 128, 128, (0, 128, 256, 384)), (501, 128, 128, (0, 128, 256, 373)),
])
def test_window_starts_include_volume_edges(length, size, stride, expected):
    assert axis_starts(length, size, stride) == expected


@pytest.mark.parametrize('shape,patch,stride', [
    ((1, 2, 3), (4, 5, 6), (4, 5, 6)), ((6, 8, 10), (3, 4, 5), (3, 4, 5)),
    ((7, 11, 13), (4, 6, 8), (3, 4, 5)), ((65, 9, 7), (64, 8, 8), (32, 4, 4)),
])
def test_crop_padding_valid_mask_and_coverage(shape, patch, stride):
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 2048
    config = PatchConfig(patch_size=patch, stride=stride)
    plan = PatchPlan(shape, config)
    counts = np.zeros(shape, dtype=np.uint32)
    expected = np.zeros(shape, dtype=np.float64)
    locations = [plan[i] for i in range(len(plan))]
    assert len({loc.start_zyx for loc in locations}) == len(plan)
    for location in locations:
        crop = extract_patch(source, location, config)
        valid = valid_region_mask(location)
        assert crop.shape == valid.shape == patch and valid.dtype == bool
        assert valid.sum() == np.prod(location.valid_shape)
        np.testing.assert_array_equal(crop[location.valid_slices], source[location.source_slices])
        assert crop.dtype == np.float32 and not np.shares_memory(crop, source)
        np.testing.assert_array_equal(crop, np.pad(source[location.source_slices], location.padding, mode='edge'))
        counts[location.source_slices] += 1
        expected[location.source_slices] += crop[location.valid_slices]
    assert np.all(counts >= 1)
    np.testing.assert_array_equal(expected / counts, source)


def test_constant_padding_does_not_change_real_voxels():
    config = PatchConfig(patch_size=(4, 5, 6), padding_mode='constant', padding_value=-1234)
    source = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    location = PatchPlan(source.shape, config)[0]
    crop = extract_patch(source, location, config)
    valid = valid_region_mask(location)
    np.testing.assert_array_equal(crop[location.valid_slices], source)
    assert np.all(crop[~valid] == -1234)
    label = extract_patch(np.ones(source.shape, dtype=np.uint8), location, config, is_mask=True)
    assert np.all(label[~valid] == 0)


def test_native_scaled_ct_and_3d_gt_are_aligned_and_files_unchanged(nifti_case):
    shape = (7, 9, 5)
    labels = (np.arange(np.prod(shape)).reshape(shape) % 3 == 0).astype(np.uint8)
    main, raw, _ = nifti_case(shape=shape, affine=np.diag([-0.7, -0.8, -5.0, 1]), labels=labels)
    image = nib.load(main.image)
    image.header.set_slope_inter(2.5, -1000)
    nib.save(image, main.image)
    files = [main.image, main.infection_mask]
    before = [(hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in files]
    config = PatchConfig(patch_size=(4, 6, 5), stride=(2, 4, 3))
    dataset = UNETRPPPatchDataset([source_case(main)], config, require_mask=True)
    assert dataset._cached is None  # Headers only until __getitem__.
    expected = (raw * 2.5 - 1000).transpose(2, 1, 0)[::-1]
    expected_mask = labels.transpose(2, 1, 0)[::-1]
    for i in range(len(dataset)):
        sample, loc = dataset[i], dataset.plans[0][i]
        assert sample['image_patch'].dtype == sample['mask_patch'].dtype == np.float32
        assert sample['valid_region'].dtype == bool
        np.testing.assert_array_equal(sample['image_patch'][0][loc.valid_slices], expected[loc.source_slices])
        np.testing.assert_array_equal(sample['mask_patch'][0][loc.valid_slices], expected_mask[loc.source_slices])
        assert sample['metadata']['canonical_orientation'] == ['L', 'P', 'S']
        assert sample['metadata']['original_orientation'] == ['L', 'P', 'I']
        assert sample['metadata']['nifti_scaling'] == {'slope': 2.5, 'intercept': -1000.0}
        json.dumps(sample['metadata'], allow_nan=False)
    after = [(hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in files]
    assert before == after


def test_inference_never_opens_gt_even_when_a_mask_path_is_supplied(nifti_case, monkeypatch):
    main, _, _ = nifti_case()
    real_load, attempted = nib.load, []
    def guarded(path, *args, **kwargs):
        assert str(path) != str(main.infection_mask), 'Inference attempted to open GT'
        attempted.append(str(path))
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(nib, 'load', guarded)
    dataset = UNETRPPPatchDataset([source_case(main)], PatchConfig(patch_size=(6, 8, 6)))
    sample = dataset[0]
    assert dataset.cases[0].mask is None and 'mask_patch' not in sample
    assert sample['metadata']['ground_truth_read'] is False
    assert 'native_lesion_voxels' not in sample['metadata']
    assert attempted and all(p == str(main.image) for p in attempted)


def test_mask_content_never_selects_or_drops_patches(nifti_case):
    first, _, _ = nifti_case(case_id='empty', labels=np.zeros((6, 8, 4), dtype=np.uint8))
    second, _, _ = nifti_case(case_id='positive', labels=np.ones((6, 8, 4), dtype=np.uint8))
    config = PatchConfig(patch_size=(3, 5, 4), stride=(2, 3, 3))
    a = UNETRPPPatchDataset([source_case(first)], config, require_mask=True)
    b = UNETRPPPatchDataset([source_case(second)], config, require_mask=True)
    assert len(a) == len(b)
    assert [a.plans[0][i].to_dict() for i in range(len(a))] == [b.plans[0][i].to_dict() for i in range(len(b))]


def test_cache_is_bounded_and_not_sent_to_windows_workers(nifti_case, monkeypatch):
    import merit_covid.data.unetrpp_input.dataset as module
    a, _, _ = nifti_case(case_id='a')
    b, _, _ = nifti_case(case_id='b')
    calls, original = [], module.read_arrays
    def traced(*args, **kwargs):
        calls.append(args[0].case_id)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'read_arrays', traced)
    dataset = UNETRPPPatchDataset([source_case(a), source_case(b)], PatchConfig(patch_size=(3, 4, 4)))
    dataset[0]; dataset[1]
    assert calls == ['a']
    dataset[dataset.ends[1]]
    assert calls == ['a', 'b'] and dataset._cached_index == 1
    assert dataset.__getstate__()['_cached'] is None


@pytest.mark.parametrize('workers', [0, 1])
def test_patch_dataloader_batches_and_metadata(nifti_case, workers):
    main, _, _ = nifti_case()
    dataset = UNETRPPPatchDataset([source_case(main)], PatchConfig(patch_size=(3, 4, 4)), require_mask=True)
    batch = next(iter(make_patch_dataloader(dataset, batch_size=2, num_workers=workers)))
    assert batch['image_patch'].shape == batch['mask_patch'].shape == batch['valid_region'].shape == (2, 1, 3, 4, 4)
    assert batch['image_patch'].dtype == torch.float32 and batch['valid_region'].dtype == torch.bool
    assert batch['patch_index'].tolist() == [0, 1] and len(batch['metadata']) == 2


@pytest.fixture
def split_manifest(nifti_case, tmp_path):
    cases = [source_case(nifti_case(case_id=f'c{i}')[0]) for i in range(10)]
    manifest, split_path = tmp_path / 'cases.json', tmp_path / 'split.json'
    manifest.write_text(json.dumps([{'id': c.case_id, 'image': str(c.image), 'mask': str(c.mask)} for c in cases]), encoding='utf-8')
    split = create_splits(cases, {'patient_ids_available': False})
    split_path.write_text(json.dumps(split), encoding='utf-8')
    return manifest, split_path, split, cases


def test_reads_existing_split_without_requiring_gt_files(split_manifest, monkeypatch):
    manifest, split_path, split, cases = split_manifest
    before = split_path.read_bytes()
    expected = split['folds'][0]['train']['case_ids']
    for case in cases:
        case.mask.unlink()
    dataset = UNETRPPPatchDataset.from_split(manifest, split_path, 0, 'train')
    assert [c.case_id for c in dataset.cases] == expected
    assert dataset[0]['image_patch'].shape == (1, 64, 128, 128)
    assert split_path.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        UNETRPPPatchDataset.from_split(manifest, split_path, require_mask=True)
    outside = split['folds'][0]['test']['case_ids'][0]
    with pytest.raises(ValueError, match='subset'):
        UNETRPPPatchDataset.from_split(manifest, split_path, case_ids=[outside])


def test_stale_split_and_invalid_fold_are_rejected(split_manifest):
    manifest, path, split, _ = split_manifest
    split['manifest_fingerprint'] = 'wrong'
    path.write_text(json.dumps(split), encoding='utf-8')
    with pytest.raises(ValueError, match='does not match'):
        UNETRPPPatchDataset.from_split(manifest, path)
    for fold in (True, -1, 5):
        with pytest.raises(ValueError):
            UNETRPPPatchDataset.from_split(manifest, path, fold)


@pytest.mark.parametrize('affine', [
    np.diag([-0.7, -0.8, 5.0, 1.0]), np.diag([-0.7, -0.8, -5.0, 1.0]),
    np.array([[0, 0, 0.7, 4], [-0.8, 0, 0, 5], [0, -5, 0, 6], [0, 0, 0, 1]]),
    np.array([[-0.7, -0.1, 0, 4], [0, -0.8, 0.2, 5], [0, 0, 5, 6], [0, 0, 0, 1]]),
])
def test_overlap_roundtrip_padding_and_restore_to_original_geometry(nifti_case, affine):
    main, raw, _ = nifti_case(shape=(7, 9, 5), affine=affine)
    config = PatchConfig(patch_size=(8, 6, 4), stride=(4, 3, 2))
    dataset = UNETRPPPatchDataset([source_case(main)], config)
    # Synthetic probability field to test geometry; not a CT normalization policy.
    native, _, _ = read_arrays(replace(source_case(main), mask=None), PreprocessConfig(), require_mask=False)
    field = (native / np.float32(raw.size)).astype(np.float32)
    assembly = PatchReconstructor(main.case_id, dataset.plans[0], dataset[0]['metadata'])
    for i in reversed(range(len(dataset))):
        loc = dataset.plans[0][i]
        patch = extract_patch(field, loc, config)
        patch[~valid_region_mask(loc)] = 0.99  # Padding must not enter the final volume.
        assembly.add(i, patch, metadata=dataset[i]['metadata'])
    prediction = assembly.finish()
    assert prediction.probability_zyx.dtype == np.float32
    np.testing.assert_allclose(prediction.probability_zyx, field, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(prediction.to_original_xyz(), raw / raw.size, rtol=1e-6, atol=1e-7)
    assert assembly.coverage_summary()['uncovered_voxels'] == 0
    assert assembly.coverage_summary()['received_patches'] == len(dataset)
    assert assembly.finish() is prediction


def test_uniform_mean_is_mean_of_probabilities(nifti_case):
    main, _, _ = nifti_case(shape=(6, 8, 4), affine=np.diag([-1, -1, 1, 1]))
    config = PatchConfig(patch_size=(3, 5, 4), stride=(2, 3, 2))
    dataset = UNETRPPPatchDataset([source_case(main)], config)
    plan = dataset.plans[0]
    assembly = PatchReconstructor(main.case_id, plan, dataset[0]['metadata'])
    numerator, denominator = np.zeros(plan.shape_zyx), np.zeros(plan.shape_zyx)
    for i in range(len(plan)):
        value = (i + 1) / (len(plan) + 1)
        assembly.add(i, np.full(config.patch_size, value, dtype=np.float32))
        numerator[plan[i].source_slices] += value
        denominator[plan[i].source_slices] += 1
    np.testing.assert_allclose(assembly.finish().probability_zyx, numerator / denominator, rtol=1e-6)
    assert assembly.coverage_summary()['maximum_contributions'] > 1


def test_reconstruction_rejects_missing_duplicate_wrong_case_and_geometry(nifti_case):
    main, _, _ = nifti_case()
    dataset = UNETRPPPatchDataset([source_case(main)], PatchConfig(patch_size=(3, 4, 4)))
    metadata = dataset[0]['metadata']
    assembly = PatchReconstructor(main.case_id, dataset.plans[0], metadata)
    with pytest.raises(ValueError, match='Missing'):
        assembly.finish()
    patch = np.full((3, 4, 4), 0.5, dtype=np.float32)
    for change in ('case_id', 'patch', 'geometry', 'input_policy'):
        bad = deepcopy(metadata)
        if change == 'case_id': bad['case_id'] = 'different'
        if change == 'patch': bad['patch']['start_zyx'][0] += 1
        if change == 'geometry': bad['canonical_affine_mm'][0][3] += 1
        if change == 'input_policy': bad['input_policy']['normalization'] = 'unexpected'
        with pytest.raises(ValueError, match='disagree'):
            assembly.add(0, patch, metadata=bad)
    for values in (patch[:, :-1], np.full_like(patch, float('nan')), np.full_like(patch, 1.01)):
        with pytest.raises(ValueError): assembly.add(0, values)
    assembly.add(0, patch, metadata=metadata)
    with pytest.raises(ValueError, match='Duplicate'): assembly.add(0, patch)
    for index in (-1, len(dataset), True, 0.5):
        with pytest.raises(IndexError): assembly.add(index, patch)


class IdentityLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, image):
        return image + self.offset


def test_full_case_input_model_and_output_integration(nifti_case):
    main, raw, _ = nifti_case(shape=(7, 9, 5), affine=np.diag([-1, -1, -2, 1]))
    dataset = UNETRPPPatchDataset([source_case(main)], PatchConfig(patch_size=(4, 6, 4), overlap=(0.5, 0.5, 0.5)))
    model = IdentityLogitModel().eval()
    before = model.offset.detach().clone()
    prediction = predict_patch_case(model, dataset, batch_size=3, model_name='synthetic identity-logit check')
    assert isinstance(prediction, VolumePrediction)
    expected = binary_logits_to_probabilities(torch.from_numpy(raw.transpose(2, 1, 0)[::-1].copy())[None, None])[0]
    np.testing.assert_allclose(prediction.probability_zyx, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(before, model.offset) and model.offset.grad is None
    policy = prediction.metadata['prediction_policy']
    assert policy['coverage']['uncovered_voxels'] == 0 and policy['activation'] == 'sigmoid'
    assert policy['output']['threshold_applied'] is False


def test_unetrpp_forward_only_with_padded_synthetic_ct(nifti_case, cpu_threads):
    pytest.importorskip('monai')
    from merit_covid.models.unetrpp import UNETRPP
    main, _, _ = nifti_case(shape=(7, 9, 5), affine=np.diag([-1, -1, 2, 1]))
    dataset = UNETRPPPatchDataset([source_case(main)], PatchConfig(patch_size=(32, 32, 32)))
    model = UNETRPP(img_size=(32, 32, 32), feature_size=2, do_ds=True).eval()
    before = {name: t.clone() for name, t in model.state_dict().items()}
    prediction = predict_patch_case(model, dataset)
    assert prediction.probability_zyx.shape == (5, 9, 7)
    assert np.isfinite(prediction.probability_zyx).all()
    assert prediction.to_original_xyz().shape == (7, 9, 5)
    for name, t in model.state_dict().items(): assert torch.equal(before[name], t), name
    assert all(p.grad is None for p in model.parameters())


def test_prediction_refuses_training_mode_or_gt_reading_dataset(nifti_case):
    main, _, _ = nifti_case()
    labeled = UNETRPPPatchDataset([source_case(main)], require_mask=True)
    with pytest.raises(ValueError, match='require_mask=False'):
        predict_patch_case(IdentityLogitModel().eval(), labeled)
    with pytest.raises(ValueError, match='model.eval'):
        predict_patch_batch(IdentityLogitModel(), {'image_patch': torch.zeros(1, 1, 2, 3, 4)})
