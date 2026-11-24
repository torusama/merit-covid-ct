"""Correctness gates for experimental B3A, not production model tests."""
from copy import deepcopy
from pathlib import Path
import pickle

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataloader import VolumeShuffleSampler, collate_slabs, seed_worker
from merit_covid.data.pipeline.geometry import restore_volume
from tools import b3a_delivery as b
from tools.qc_tiling128 import TILES, stitch_tiles


def make_case(root, name, depth, affine, lesion='varying'):
    root.mkdir(parents=True, exist_ok=True)
    z, y, x = np.ogrid[:depth, :512, :512]
    image = (z * .17 + y * .0013 - x * .0021 - .3).astype(np.float32)
    image[0, 0, :2] = [0., -0.]
    mask = np.zeros(image.shape, dtype=np.uint8)
    if lesion == 'tiny':
        mask[-1, 511, 511] = 1
    elif lesion == 'varying':
        mask[:, 127:129, 127:129] = 1
        for k in range(depth):
            mask[k, 12 + k, 300 + k] = 1
    case = Case(name, root / (name + '_ct.nii.gz'), root / (name + '_seg.nii'), name)
    for path, array in ((case.image, image), (case.mask, mask)):
        obj = nib.Nifti1Image(array.transpose(2, 1, 0), affine)
        obj.header.set_xyzt_units('mm')
        obj.set_sform(affine, code=1)
        try:
            obj.set_qform(affine, code=2, strip_shears=False)
        except nib.spatialimages.HeaderDataError:
            obj.set_qform(None, code=0)
        nib.save(obj, path)
    return case


@pytest.fixture(scope='module')
def fixture_export(tmp_path_factory):
    root = tmp_path_factory.mktemp('b3a_synthetic')
    theta = np.deg2rad(23)
    rotate = np.array([[np.cos(theta), 0, np.sin(theta), 31],
                       [0, 1, 0, -72], [-np.sin(theta), 0, np.cos(theta), 5], [0, 0, 0, 1]])
    cases = [make_case(root / 'source', 'z1_empty', 1, np.diag([.7, .8, 2., 1]), 'empty'),
             make_case(root / 'source', 'z2_tiny_flip', 2, np.diag([-.7, .8, -2., 1]), 'tiny'),
             make_case(root / 'source', 'z13_oblique', 13, rotate @ np.diag([-.7, -.8, 2.5, 1]))]
    config = PreprocessConfig()
    manifest = b.export_volume_tiles(cases, config, {c.case_id: 'train' for c in cases}, root / 'export', provenance={'synthetic_fixture': True})
    return cases, config, manifest


@pytest.mark.parametrize('case_index', [0, 1, 2])
def test_all_sample_values_geometry_dtype_and_context(fixture_export, case_index):
    cases, config, manifest = fixture_export
    case = cases[case_index]
    a = b.TileDeliveryDataset([case], config, mode='dynamic', order='case_tile_master')
    c = b.TileDeliveryDataset([case], config, mode='precomputed', manifest=manifest, order='case_tile_master')
    full, gt, geometry = b.read_arrays(case, config)
    assert len(a) == len(c) == 16 * len(full)
    for index in range(len(a)):
        left, right = a[index], c[index]
        b.compare_samples(left, right)
        key = a.key_for_index(index)
        tile = TILES[key.tile_id]
        selected = np.clip(np.arange(key.master_z - 5, key.master_z + 6), 0, len(full) - 1)
        expected_ct = full[selected, tile.y0:tile.y1, tile.x0:tile.x1]
        expected_gt = gt[selected, tile.y0:tile.y1, tile.x0:tile.x1]
        np.testing.assert_array_equal(left['image_slab'][0].numpy().view(np.uint32), expected_ct.view(np.uint32))
        np.testing.assert_array_equal(left['mask_slab'][0].numpy(), expected_gt)
        np.testing.assert_array_equal(left['master_mask'][0].numpy(), gt[key.master_z, tile.y0:tile.y1, tile.x0:tile.x1])
        centre = np.array([tile.x0 + 63.5, tile.y0 + 63.5, key.master_z, 1])
        expected_world = np.asarray(geometry['canonical_affine_mm']) @ centre
        np.testing.assert_allclose(left['slice_positions_mm'][5], expected_world[:3], atol=1e-8, rtol=0)
        assert left['metadata']['original_shape_xyz'] == geometry['original_shape_xyz']
        assert left['metadata']['tile_geometry']['canonical_shape_xyz'][:2] == [128, 128]
        assert left['metadata']['full_volume_canonical_shape_xyz'][:2] == [512, 512]


def test_translated_tile_world_centre_is_not_full_image_centre(fixture_export):
    cases, config, manifest = fixture_export
    data = b.TileDeliveryDataset([cases[2]], config, mode='precomputed', manifest=manifest)
    sample = data[0]
    full_affine = np.asarray(sample['metadata']['full_volume_canonical_affine_mm'])
    full_centre = (full_affine @ [255.5, 255.5, 0, 1])[:3]
    assert not np.allclose(sample['slice_positions_mm'][5], full_centre)
    np.testing.assert_array_equal(sample['metadata']['original_affine_mm'], sample['metadata']['full_geometry']['original_affine_mm'])


@pytest.mark.parametrize('case_index', [0, 1, 2])
def test_full_tile_reconstruction_and_original_restore(fixture_export, case_index):
    cases, config, path = fixture_export
    manifest = b.validate_export(path)
    case = cases[case_index]
    record = next(c for c in manifest['source_cases'] if c['case_id'] == case.case_id)
    ct = stitch_tiles((TILES[t['tile_id']], np.load(path.parent / t['ct']['path'])) for t in record['tiles'])
    gt = stitch_tiles((TILES[t['tile_id']], np.load(path.parent / t['gt']['path'])) for t in record['tiles'])
    original, truth, full = b.read_arrays(case, config)
    np.testing.assert_array_equal(ct.view(np.uint32), original.view(np.uint32))
    np.testing.assert_array_equal(gt, truth)
    restored = restore_volume(gt, full)
    np.testing.assert_array_equal(restored, np.asanyarray(nib.load(case.mask).dataobj))
    source = nib.load(case.image).get_fdata(dtype=np.float32)
    restored_ct = nib.orientations.apply_orientation(ct.transpose(2, 1, 0), np.asarray(full['canonical_to_original_orientation']))
    np.testing.assert_array_equal(restored_ct.view(np.uint32), source.view(np.uint32))


def test_qform_sform_provenance_and_shear(tmp_path):
    affine = np.array([[-.7, .09, .03, 102], [0, -.8, .12, -73], [0, 0, 2.5, 11], [0, 0, 0, 1]])
    case = make_case(tmp_path / 'input', 'shear', 2, affine)
    path = b.export_volume_tiles([case], PreprocessConfig(), {'shear': 'train'}, tmp_path / 'tiles')
    record = b.validate_export(path)['source_cases'][0]
    assert record['source_nifti_forms']['ct']['qform_code'] == 0
    assert record['source_nifti_forms']['ct']['sform_code'] == 1
    data = b.TileDeliveryDataset([case], PreprocessConfig(), mode='precomputed', manifest=path)
    sample = data[15]
    f = np.asarray(sample['metadata']['full_volume_canonical_affine_mm'])
    shift = np.eye(4); shift[:3, 3] = [384, 384, 0]
    np.testing.assert_array_equal(sample['metadata']['tile_affine_mm'], f @ shift)
    a = b.TileDeliveryDataset([case], PreprocessConfig(), mode='dynamic')
    b.compare_samples(a[15], sample)


def test_ct_only_never_opens_gt(tmp_path, monkeypatch):
    case = make_case(tmp_path / 'input', 'ctonly', 2, np.diag([-.7, -.8, 2, 1]))
    config = PreprocessConfig()
    path = b.export_volume_tiles([case], config, {'ctonly': 'train'}, tmp_path / 'export')
    load = nib.load
    def ct_load(filename, *args, **kwargs):
        if 'seg' in str(filename):
            raise AssertionError('GT opened in CT-only path')
        return load(filename, *args, **kwargs)
    monkeypatch.setattr(nib, 'load', ct_load)
    for tile in b.validate_export(path)['source_cases'][0]['tiles']:
        (path.parent / tile['gt']['path']).unlink()
    b.validate_export(path, require_mask=False)
    a = b.TileDeliveryDataset([case], config, mode='dynamic', require_mask=False)
    c = b.TileDeliveryDataset([case], config, mode='precomputed', manifest=path, require_mask=False)
    left, right = a[0], c[0]
    assert 'mask_slab' not in left and 'master_mask' not in left
    b.compare_samples(left, right)


def test_original_volume_groups_lengths_sampler_and_pickle(fixture_export):
    cases, config, manifest = fixture_export
    a = b.TileDeliveryDataset(cases, config, mode='dynamic')
    c = b.TileDeliveryDataset(cases, config, mode='precomputed', manifest=manifest)
    assert a.ends == c.ends == [0, 16, 48, 256]
    assert len(a.cases) == len(c.cases) == 3  # not48 new volume/patient identities
    indices_a = list(VolumeShuffleSampler(a, torch.Generator().manual_seed(42)))
    indices_b = list(VolumeShuffleSampler(c, torch.Generator().manual_seed(42)))
    assert indices_a == indices_b and len(set(indices_a)) == len(a)
    for index in indices_a:
        key = a.key_for_index(index)
        assert key == c.key_for_index(index)
        assert a.index_for_key(key) == c.index_for_key(key) == index
    a[0]
    state = pickle.loads(pickle.dumps(a))
    assert state._cached is None and state._cached_index is None


@pytest.mark.parametrize('order', ['case_master_tile', 'case_tile_master'])
@pytest.mark.parametrize('workers', [0, 2])
def test_dataloader_collate_order_and_consumed_cursor_resume(fixture_export, workers, order):
    cases, config, manifest = fixture_export
    # Tiny Z1 case keeps this a worker/contract test, never a benchmark.
    a = b.TileDeliveryDataset(cases[:1], config, mode='dynamic', order=order)
    c = b.TileDeliveryDataset(cases[:1], config, mode='precomputed', manifest=manifest, order=order)
    indices = list(range(8))
    def batches(dataset, start=0):
        loader = DataLoader(dataset, batch_size=2, sampler=b.FixedItinerary(indices, start=start),
            collate_fn=collate_slabs, num_workers=workers, pin_memory=False,
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))
        return list(loader)
    full_a, full_b = batches(a), batches(c)
    for left, right in zip(full_a, full_b):
        assert left['image_slab'].shape == (2, 1, 11, 128, 128)
        b.compare_samples(left, right)
    for left, right in zip(full_b[2:], batches(c, start=4)):
        b.compare_samples(left, right)


def test_atomic_failure_incomplete_detection_and_no_overwrite(tmp_path, monkeypatch):
    case = make_case(tmp_path / 'input', 'atomic', 1, np.diag([-.7, -.8, 2, 1]))
    original = b.atomic_npy
    calls = []
    def fail(path, array):
        calls.append(path)
        if len(calls) == 2:
            raise OSError('injected interrupted export')
        return original(path, array)
    monkeypatch.setattr(b, 'atomic_npy', fail)
    output = tmp_path / 'broken'
    with pytest.raises(OSError, match='interrupted'):
        b.export_volume_tiles([case], PreprocessConfig(), {'atomic': 'train'}, output)
    assert not output.exists()
    incomplete = list(tmp_path.glob('broken.incomplete-*'))
    assert len(incomplete) == 1 and (incomplete[0] / 'EXPORT_ERROR.json').is_file()
    with pytest.raises(FileExistsError, match='Incomplete export'):
        b.export_volume_tiles([case], PreprocessConfig(), {'atomic': 'train'}, output)


def test_hash_validation_detects_corruption_and_existing_export_refused(tmp_path):
    case = make_case(tmp_path / 'input', 'corrupt', 1, np.diag([-.7, -.8, 2, 1]))
    output = tmp_path / 'export'
    path = b.export_volume_tiles([case], PreprocessConfig(), {'corrupt': 'train'}, output)
    with pytest.raises(FileExistsError, match='no overwrite'):
        b.export_volume_tiles([case], PreprocessConfig(), {'corrupt': 'train'}, output)
    item = b.validate_export(path)['source_cases'][0]['tiles'][0]['ct']
    with (path.parent / item['path']).open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='corrupted tile'):
        b.validate_export(path)


def test_runner_materializes_same_full_payload_and_itinerary(fixture_export):
    from tools.b3a_compare import run_pass
    torch.set_num_threads(1)
    cases, config, manifest = fixture_export
    a = b.TileDeliveryDataset(cases[:1], config, mode='dynamic')
    c = b.TileDeliveryDataset(cases[:1], config, mode='precomputed', manifest=manifest)
    left = run_pass(a, list(range(16)), 0, 2, 'unit-A')
    right = run_pass(c, list(range(16)), 0, 2, 'unit-B')
    assert left['samples'] == right['samples'] == 16
    assert left['batches'] == right['batches'] == 8
    assert left['itinerary_sha256'] == right['itinerary_sha256']
    assert left['consumer_checksum'] == right['consumer_checksum']
    assert left['ct_payload_bytes'] == right['ct_payload_bytes'] == 16 * 11 * 128 * 128 * 4
    assert left['gt_payload_bytes'] == right['gt_payload_bytes'] == 16 * 12 * 128 * 128 * 4
    assert left['rss_peak_main_plus_workers_bytes'] > 0


def test_runner_pilot_selection_is_train_only_and_deterministic(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from tools.b3a_compare import select_pilot
    names = ['volume-covid19-A-0003', *[f'case{i:02}' for i in range(30)]]
    cases = [Case(n, tmp_path / (n + '_ct.nii'), tmp_path / (n + '_gt.nii'), n) for n in names]
    entries = {c.case_id: dict(shape=[512, 512, i + 30], gt_lesion_voxels=i * 17) for i, c in enumerate(cases)}
    parts = {c.case_id: ('train' if i < 25 else 'test') for i, c in enumerate(cases)}
    monkeypatch.setattr(nib, 'load', lambda _: SimpleNamespace(header=SimpleNamespace(get_zooms=lambda: (1., 1., 5.))))
    first, second = select_pilot(cases, entries, parts), select_pilot(cases, entries, parts)
    assert [c.case_id for c in first] == [c.case_id for c in second]
    assert len(first) == len({c.case_id for c in first}) == 13
    assert all(parts[c.case_id] == 'train' for c in first)
    assert 'volume-covid19-A-0003' in {c.case_id for c in first}


def test_dynamic_does_not_materialize_full512_slab_per_tile(fixture_export, monkeypatch):
    cases, config, manifest = fixture_export
    original = b.sample_arrays
    shapes = []
    def observed(image, mask, *args, **kwargs):
        shapes.append(image.shape[-2:])
        return original(image, mask, *args, **kwargs)
    monkeypatch.setattr(b, 'sample_arrays', observed)
    a = b.TileDeliveryDataset(cases[:1], config, mode='dynamic')
    c = b.TileDeliveryDataset(cases[:1], config, mode='precomputed', manifest=manifest)
    for index in (0, 5, 15):
        b.compare_samples(a[index], c[index])
    assert shapes == [(128, 128)] * 6
