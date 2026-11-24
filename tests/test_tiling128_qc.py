"""B2 QC tests: exact pixels, coordinate axes, real context GT and honest gates."""
import gzip
import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from tools import qc_tiling128 as qc
from merit_covid.data.pipeline.cases import Case, manifest_fingerprint
from merit_covid.data.pipeline.splits import create_splits


def arrays(depth=3):
    z, y, x = np.ogrid[:depth, :512, :512]
    image = (z * 1_000_000 + y * 512 + x).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    for index in range(depth):
        mask[index, (index * 41) % 512, (index * 73) % 512] = 1
    metadata = dict(canonical_shape_xyz=[512, 512, depth],
                    canonical_affine_mm=np.diag([-0.7, -0.8, 5.0, 1]).tolist())
    return image, mask, metadata


def test_one_tile_row_is_y_and_col_is_x():
    image, _, _ = arrays(1)
    tile = qc.TILES[2 * 4 + 1]
    assert (tile.tile_row, tile.tile_col) == (2, 1)
    assert (tile.x0, tile.x1, tile.y0, tile.y1) == (128, 256, 256, 384)
    part = qc.crop_tile(image[0], tile)
    assert part.shape == (128, 128)
    assert part[0, 0] == 256 * 512 + 128
    assert part[-1, -1] == 383 * 512 + 255


def test_sixteen_tiles_roundtrip_arbitrary_order_and_dtype():
    image, mask, _ = arrays(2)
    for value in (image, mask):
        result = qc.stitch_tiles((t, qc.crop_tile(value, t)) for t in reversed(qc.TILES))
        assert result.dtype == value.dtype
        np.testing.assert_array_equal(result, value)


def test_lesion_crosses_four_tile_boundaries_without_loss():
    image, mask, metadata = arrays(1)
    mask[:] = 0
    mask[0, 127:129, 127:129] = 1
    report = qc.qc_arrays(image, mask, metadata, case_id='cross', partition='train')
    assert report['tiles_lesion'] == 4
    assert report['tiles_empty'] == 12
    assert report['gt_lesion_before'] == report['gt_lesion_after'] == 4


def test_single_pixel_and_metadata_keep_last_xy_position():
    image, mask, metadata = arrays(1)
    mask[:] = 0
    mask[0, 511, 511] = 1
    stream = io.StringIO()
    report = qc.qc_arrays(image, mask, metadata, case_id='tiny', partition='test', tile_stream=stream)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert report['tiles_lesion'] == 1
    assert report['gt_lesion_before'] == report['gt_lesion_after'] == 1
    assert len(records) == 16
    assert records[-1] == dict(case_id='tiny', partition='test', master_z=0,
        tile_row=3, tile_col=3, x0=384, x1=512, y0=384, y1=512, lesion_pixels=1)


def test_float32_intensity_bit_patterns_survive():
    rng = np.random.default_rng(42)
    image = rng.normal(-1.3, 0.8, (512, 512)).astype(np.float32)
    image[0, :4] = [0.0, -0.0, np.nextafter(np.float32(0), np.float32(1)), -1.234567]
    restored = qc.stitch_tiles((t, qc.crop_tile(image, t)) for t in qc.TILES)
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored.view(np.uint32), image.view(np.uint32))


@pytest.mark.parametrize('label', [2, -1, 0.5, 1 + 1e-10])
def test_nonbinary_gt_is_rejected(label):
    image, mask, metadata = arrays(1)
    mask = mask.astype(np.float64)
    mask[0, 2, 3] = label
    with pytest.raises(ValueError, match='exact binary'):
        qc.qc_arrays(image, mask, metadata, case_id='bad', partition='train')


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_ct_is_rejected(value):
    image, mask, metadata = arrays(1)
    image[0, 2, 3] = value
    with pytest.raises(ValueError, match='nonfinite'):
        qc.qc_arrays(image, mask, metadata, case_id='bad', partition='train')


@pytest.mark.parametrize('depth', [1, 2, 3, 11, 13])
def test_slab11_edge_unique_and_actual_context_gt(depth):
    image, mask, metadata = arrays(depth)
    rows = qc.check_slabs(image, mask, metadata)
    assert rows[0]['slice_indices'][5] == 0
    assert rows[-1]['slice_indices'][5] == depth - 1
    assert rows[0]['slice_indices'][:6] == [0] * 6
    for row in rows:
        indices, valid = np.array(row['slice_indices']), np.array(row['slice_valid'])
        assert valid[5] and len(set(indices[valid])) == len(indices[valid])
        assert set(indices[valid]) == set(indices)


def test_context_gt_cannot_be_repeated_master(monkeypatch):
    image, mask, metadata = arrays(13)
    original = qc.sample_arrays

    def broken(*args, **kwargs):
        sample = original(*args, **kwargs)
        sample['mask_slab'] = np.repeat(sample['master_mask'][:, None], 11, axis=1)
        return sample

    monkeypatch.setattr(qc, 'sample_arrays', broken)
    with pytest.raises(ValueError, match='Slab GT Z/XY mismatch'):
        qc.check_slabs(image, mask, metadata)


def test_duplicate_missing_or_wrong_shape_tiles_fail():
    image, _, _ = arrays(1)
    pairs = [(t, qc.crop_tile(image[0], t)) for t in qc.TILES]
    with pytest.raises(ValueError, match='16 unique'):
        qc.stitch_tiles(pairs[:-1] + pairs[:1])
    with pytest.raises(ValueError, match='16 unique'):
        qc.stitch_tiles(pairs[:-1])
    with pytest.raises(ValueError, match='shapes/dtypes'):
        qc.stitch_tiles([(pairs[0][0], pairs[0][1].astype(np.float64)), *pairs[1:]])


def test_existing_split_detects_volume_leakage(tmp_path):
    cases = [Case(f'c{i}', tmp_path / f'c{i}_ct.nii', tmp_path / f'c{i}_seg.nii', f'c{i}') for i in range(199)]
    split = create_splits(cases, {'group_source': 'volume_id_fallback', 'patient_ids_available': False})
    split['folds'][0]['validation']['case_ids'][0] = split['folds'][0]['train']['case_ids'][0]
    split['folds'][0]['validation']['group_ids'][0] = split['folds'][0]['train']['group_ids'][0]
    with pytest.raises(ValueError, match='leakage'):
        qc.validate_splits(split, cases)


def test_missing_export_produces_blocked_report(tmp_path):
    root = tmp_path / 'dataset'
    root.mkdir()
    output = tmp_path / 'report'
    summary = qc.run_qc(data_root=root, output_dir=output)
    assert summary['status'] == 'BLOCKED_INPUT'
    assert summary['tested_cases'] == 0 and not summary['b2_gate_pass']
    assert 'Missing precomputed V3 inputs' in summary['failures'][0]['reason']
    assert (output / 'tiling128_qc_199.csv').is_file()
    with pytest.raises(FileExistsError):
        qc.run_qc(data_root=root, output_dir=output)


@pytest.mark.parametrize('change', [{'size_hw': [128, 128]}, {'boundary_policy': 'reflect'},
                                    {'duplicate_slice_policy': 'all'}, {'intensity_encoding': 'normalized_0_1'}])
def test_precomputed_policy_forbids_resize_or_reprocessing(tmp_path, change):
    for name in ('cases.json', 'splits_covid19_20_5fold.json', 'EXPORT_COMPLETE.json'):
        (tmp_path / name).write_text('{}', encoding='utf-8')
    config = dict(intensity_encoding='native', size_hw=None, slab_size=11,
                  boundary_policy='edge', duplicate_slice_policy='unique', **{})
    config.update(change)
    (tmp_path / 'merit_precomputed_v3.yaml').write_text(yaml.safe_dump({'preprocess': config}), encoding='utf-8')
    with pytest.raises(ValueError, match='PASS-THROUGH'):
        qc.load_inputs(tmp_path, 'cases.json', 'splits_covid19_20_5fold.json',
                       'merit_precomputed_v3.yaml', 'EXPORT_COMPLETE.json')


def test_nifti_case_read_pass_through_and_affine_failure(tmp_path):
    image, mask, _ = arrays(2)
    image = (image / np.float32(1234567) - np.float32(1.2)).astype(np.float32)
    affine = np.diag([-0.7, -0.8, 5.0, 1])
    ct = nib.Nifti1Image(image.transpose(2, 1, 0), affine)
    gt = nib.Nifti1Image(mask.transpose(2, 1, 0), affine)
    for obj in (ct, gt):
        obj.header.set_xyzt_units('mm')
    case = Case('fixture', tmp_path / 'ct.nii.gz', tmp_path / 'gt.nii', 'fixture')
    nib.save(ct, case.image)
    nib.save(gt, case.mask)
    export = dict(shape=list(ct.shape), ct_output_sha256=qc.file_hash(case.image),
                  gt_output_sha256=qc.file_hash(case.mask), gt_lesion_voxels=int(mask.sum()))
    row = qc.qc_case(case, qc.PreprocessConfig(), 'train', export, io.StringIO())
    assert row['status'] == 'PASS', row['reason']
    assert row['source_unchanged'] and row['ct_mismatched_bits'] == 0
    decoded, _, _ = qc.read_arrays(case, qc.PreprocessConfig())
    np.testing.assert_array_equal(decoded.view(np.uint32), image.view(np.uint32))
    gt.affine[0, 3] = 9
    nib.save(gt, case.mask)
    export['gt_output_sha256'] = qc.file_hash(case.mask)
    row = qc.qc_case(case, qc.PreprocessConfig(), 'train', export, io.StringIO())
    assert row['status'] == 'FAIL' and 'geometry mismatch' in row['reason']


def test_partial_selection_cannot_claim_199_pass(tmp_path, monkeypatch):
    root = tmp_path / 'input'
    root.mkdir()
    cases = [Case(f'c{i:03}', root / f'c{i:03}_ct.nii', root / f'c{i:03}_seg.nii', f'c{i:03}') for i in range(199)]
    identity = {'group_source': 'volume_id_fallback', 'patient_ids_available': False}
    split = create_splits(cases, identity)
    partitions = {c: role for role in qc.EXPECTED_PARTITIONS for c in split['folds'][0][role]['case_ids']}
    metadata = dict(status='COMPLETE_WITH_PROVENANCE_LIMITATION', policy=qc.POLICY,
        count=199, train_count=127, split_ids_unchanged=True,
        new_manifest_fingerprint=manifest_fingerprint(cases), files=[
            dict(case_id=c.case_id, partition=partitions[c.case_id], ct_output_file=c.image.name,
                 gt_output_file=c.mask.name, ct_output_sha256='0' * 64, gt_output_sha256='0' * 64)
            for c in cases])
    (root / 'cases.json').write_text('[]')
    (root / 'splits_covid19_20_5fold.json').write_text(json.dumps(split))
    (root / 'EXPORT_COMPLETE.json').write_text(json.dumps(metadata))
    (root / 'merit_precomputed_v3.yaml').write_text(yaml.safe_dump({'preprocess': {'intensity_encoding': 'native'}}))
    monkeypatch.setattr(qc, 'load_cases', lambda *a, **k: (cases, identity))
    monkeypatch.setattr(qc, 'qc_case', lambda c, p, role, e, s: dict(case_id=c.case_id, partition=role, status='PASS', reason='fixture', seconds=0))
    report = qc.run_qc(data_root=root, output_dir=tmp_path / 'report', case_ids=['c000'])
    assert report['status'] == 'PARTIAL_PASS'
    assert report['tested_cases'] == report['passed_cases'] == 1
    assert report['not_tested_cases'] == 198 and not report['b2_gate_pass']
    assert report['warnings']
