"""B2 read-only QC of 16 nonoverlapping native128 tiles from precomputed V3.

No Dataset/DataLoader/model integration, tensor export, normalization or training.
Uses the production read_arrays and edge/unique slab sampler without changing them.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import nibabel as nib
import numpy as np
import yaml

from merit_covid.data.pipeline.cases import load_cases, manifest_fingerprint
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.preprocessing import read_arrays
from merit_covid.data.pipeline.slab import sample_arrays
from merit_covid.data.pipeline.splits import validate_splits

EXPECTED_CASES = 199
EXPECTED_PARTITIONS = {'train': 127, 'validation': 32, 'test': 40}
POLICY = 'train_fovaware_clipped_zscore_v1'
TILE_SIZE = 128
GRID_SIZE = 4
FIELDS = [
    'case_id', 'partition', 'status', 'reason', 'shape_zyx', 'ct_dtype', 'gt_dtype',
    'ct_finite', 'gt_binary', 'voxel_pairs_checked', 'tiles_total', 'tiles_lesion',
    'tiles_empty', 'gt_lesion_before', 'gt_lesion_after', 'ct_mismatched_pixels',
    'ct_mismatched_bits', 'gt_mismatched_pixels', 'slab_masters', 'slab_checks_passed',
    'ct_sha256_before', 'ct_sha256_after', 'gt_sha256_before', 'gt_sha256_after',
    'hashes_match_export', 'source_unchanged', 'seconds',
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@dataclass(frozen=True)
class Tile:
    tile_row: int
    tile_col: int
    x0: int
    x1: int
    y0: int
    y1: int


TILES = tuple(Tile(row, col, col * 128, (col + 1) * 128,
                   row * 128, (row + 1) * 128)
              for row in range(4) for col in range(4))


def crop_tile(array, tile):
    """Axes end in Y,X. Slicing preserves dtype, values and Z ordering."""
    if array.ndim < 2 or array.shape[-2:] != (512, 512) or tile not in TILES:
        raise ValueError('Expected native512 Y,X and a valid fixed128 tile')
    return array[..., tile.y0:tile.y1, tile.x0:tile.x1]


def stitch_tiles(pairs):
    """Reconstruct native512 by explicit coordinates; no interpolation or casts."""
    pairs = list(pairs)
    if len(pairs) != 16 or {tile for tile, _ in pairs} != set(TILES):
        raise ValueError('Need each of the 16 unique tile coordinates exactly once')
    first = pairs[0][1]
    if first.ndim < 2:
        raise ValueError('Tiles must have Y,X axes')
    expected_shape = (*first.shape[:-2], 128, 128)
    result = np.empty((*first.shape[:-2], 512, 512), dtype=first.dtype)
    for tile, value in pairs:
        if value.shape != expected_shape or value.dtype != first.dtype:
            raise ValueError('Tile shapes/dtypes differ; implicit conversion forbidden')
        result[..., tile.y0:tile.y1, tile.x0:tile.x1] = value
    return result


def representative_masters(depth):
    if depth < 1:
        raise ValueError('Empty volume')
    return sorted({max(0, min(depth - 1, z))
                   for z in (0, 1, 4, 5, depth // 2, depth - 6, depth - 5, depth - 2, depth - 1)})


def check_slabs(image, mask, metadata):
    """Independent clipped-index checks and source GT crops for every context slot."""
    details = []
    depth = len(image)
    for master in representative_masters(depth):
        sample = sample_arrays(image, mask, master, 11, metadata,
                               sampling_mode='index', boundary_policy='edge',
                               duplicate_slice_policy='unique')
        expected = np.clip(np.arange(master - 5, master + 6), 0, depth - 1)
        if not np.array_equal(sample['slice_indices'], expected):
            raise ValueError(f'Slab master {master}: wrong edge indices')
        valid = sample['slice_valid']
        selected_valid = expected[valid]
        if (valid.shape != (11,) or not valid[5]
                or len(set(selected_valid.tolist())) != len(selected_valid)
                or set(selected_valid.tolist()) != set(expected.tolist())
                or not np.array_equal(sample['attention_valid'], valid)):
            raise ValueError(f'Slab master {master}: duplicate_slice_policy unique failed')
        if sample['image_slab'].shape != (1, 11, 512, 512) or sample['mask_slab'].shape != (1, 11, 512, 512):
            raise ValueError('Unexpected production slab shape')
        for tile in TILES:
            ct_tile = crop_tile(sample['image_slab'][0], tile)
            gt_tile = crop_tile(sample['mask_slab'][0], tile)
            for slot, source_z in enumerate(expected):
                if not np.array_equal(ct_tile[slot], image[source_z, tile.y0:tile.y1, tile.x0:tile.x1]):
                    raise ValueError(f'Slab CT Z/XY mismatch at master={master}, slot={slot}, tile={tile}')
                if not np.array_equal(gt_tile[slot], mask[source_z, tile.y0:tile.y1, tile.x0:tile.x1]):
                    raise ValueError(f'Slab GT Z/XY mismatch at master={master}, slot={slot}, tile={tile}')
            if not np.array_equal(crop_tile(sample['master_mask'][0], tile), gt_tile[5]):
                raise ValueError('GT master differs from actual centre slice')
        if not np.array_equal(stitch_tiles((t, crop_tile(sample['image_slab'][0], t)) for t in TILES), sample['image_slab'][0]):
            raise ValueError('Slab CT roundtrip failed')
        if not np.array_equal(stitch_tiles((t, crop_tile(sample['mask_slab'][0], t)) for t in TILES), sample['mask_slab'][0]):
            raise ValueError('Slab GT roundtrip failed')
        details.append(dict(master_z=master, slice_indices=expected.tolist(),
                            slice_valid=valid.tolist(),
                            unique_slices=int(valid.sum()),
                            z_offsets_mm=sample['z_offsets_mm'].tolist()))
        del sample
    return details


def qc_arrays(image, mask, metadata, *, case_id, partition, tile_stream=None):
    """Compare every voxel to reconstructed slices; retain only one slice output."""
    if (image.ndim != 3 or image.shape != mask.shape or image.shape[1:] != (512, 512)
            or len(image) < 1 or image.dtype != np.float32):
        raise ValueError('CT/GT need aligned native512 Z,Y,X; CT must be float32')
    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError('CT/GT contains nonfinite values')
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('GT must be exact binary 0/1')
    result = dict(shape_zyx=list(image.shape), ct_dtype=str(image.dtype), gt_dtype=str(mask.dtype),
                  ct_finite=True, gt_binary=True, voxel_pairs_checked=0,
                  tiles_total=0, tiles_lesion=0, tiles_empty=0,
                  gt_lesion_before=int(np.count_nonzero(mask)), gt_lesion_after=0,
                  ct_mismatched_pixels=0, ct_mismatched_bits=0, gt_mismatched_pixels=0)
    for master in range(len(image)):
        ct_pairs, gt_pairs = [], []
        for tile in TILES:
            ct_tile, gt_tile = crop_tile(image[master], tile), crop_tile(mask[master], tile)
            ct_pairs.append((tile, ct_tile))
            gt_pairs.append((tile, gt_tile))
            lesion_pixels = int(np.count_nonzero(gt_tile))
            result['tiles_total'] += 1
            result['tiles_lesion' if lesion_pixels else 'tiles_empty'] += 1
            if tile_stream is not None:
                tile_stream.write(json.dumps(dict(case_id=case_id, partition=partition,
                    master_z=master, **asdict(tile), lesion_pixels=lesion_pixels), separators=(',', ':')) + '\n')
        ct_restored, gt_restored = stitch_tiles(ct_pairs), stitch_tiles(gt_pairs)
        result['ct_mismatched_pixels'] += int(np.count_nonzero(ct_restored != image[master]))
        result['ct_mismatched_bits'] += int(np.count_nonzero(ct_restored.view(np.uint32) != image[master].view(np.uint32)))
        result['gt_mismatched_pixels'] += int(np.count_nonzero(gt_restored != mask[master]))
        result['gt_lesion_after'] += int(np.count_nonzero(gt_restored))
        result['voxel_pairs_checked'] += image[master].size
    if (result['ct_mismatched_pixels'] or result['ct_mismatched_bits']
            or result['gt_mismatched_pixels'] or result['gt_lesion_before'] != result['gt_lesion_after']):
        raise ValueError(f'CT/GT roundtrip mismatch: {result}')
    details = check_slabs(image, mask, metadata)
    result.update(slab_masters=[d['master_z'] for d in details], slab_checks_passed=len(details), slab_details=details)
    return result


def resolve_input(data_root, path):
    path = Path(path)
    return (path if path.is_absolute() else data_root / path).resolve()


def load_inputs(data_root, manifest, split, config, export):
    """Gate pass-through policy, exported hashes/IDs and volume-level splits."""
    paths = dict(manifest=resolve_input(data_root, manifest), split=resolve_input(data_root, split),
                 config=resolve_input(data_root, config), export=resolve_input(data_root, export))
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing precomputed V3 inputs: ' + ', '.join(missing)
            + '. Point --data-root at the exported V3 dataset; do not substitute raw CT.')
    cfg = yaml.safe_load(paths['config'].read_text(encoding='utf-8-sig'))
    preprocess = PreprocessConfig(**cfg['preprocess'])
    if (preprocess.intensity_encoding != 'native' or preprocess.size_hw is not None
            or preprocess.slab_size != 11 or preprocess.sampling_mode != 'index'
            or preprocess.boundary_policy != 'edge' or preprocess.duplicate_slice_policy != 'unique'):
        raise ValueError('B2 requires native PASS-THROUGH of precomputed V3, size_hw:null, slab11 index/edge/unique; no normalization or resize')
    exported = json_read(paths['export'])
    if (exported.get('status') not in ('COMPLETE', 'COMPLETE_WITH_PROVENANCE_LIMITATION')
            or exported.get('policy') != POLICY or exported.get('count') != EXPECTED_CASES
            or exported.get('train_count') != 127 or exported.get('split_ids_unchanged') is not True):
        raise ValueError('EXPORT_COMPLETE.json does not identify a completed 199-case V3 export with unchanged splits')
    # Don't require all files to exist here: missing images must appear as case failures.
    cases, identity = load_cases(paths['manifest'], expected_count=EXPECTED_CASES, check_masks=False)
    records = exported.get('files', [])
    if len(records) != EXPECTED_CASES or len({r['case_id'] for r in records}) != EXPECTED_CASES:
        raise ValueError('Export must have exactly 199 unique case records')
    by_id = {r['case_id']: r for r in records}
    if set(by_id) != {c.case_id for c in cases}:
        raise ValueError('Export and manifest IDs differ')
    for case in cases:
        if not case.image.is_relative_to(data_root) or not case.mask.is_relative_to(data_root):
            raise ValueError(f'{case.case_id}: source path escapes --data-root')
        entry = by_id[case.case_id]
        if (case.image.name != Path(entry['ct_output_file']).name
                or case.mask.name != Path(entry['gt_output_file']).name):
            raise ValueError(f'{case.case_id}: manifest filenames differ from export')
        for key in ('ct_output_sha256', 'gt_output_sha256'):
            value = entry[key]
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value.lower()):
                raise ValueError(f'{case.case_id}: invalid {key}')
    if exported.get('new_manifest_fingerprint') != manifest_fingerprint(cases):
        raise ValueError('Export fingerprint differs from current manifest/group mapping')
    split_data = json_read(paths['split'])
    validate_splits(split_data, cases)
    fold = next(f for f in split_data['folds'] if f['fold'] == 0)
    partitions = {case_id: role for role in EXPECTED_PARTITIONS for case_id in fold[role]['case_ids']}
    if {role: len(fold[role]['case_ids']) for role in EXPECTED_PARTITIONS} != EXPECTED_PARTITIONS:
        raise ValueError('Fold0 must contain exactly TRAIN127/VAL32/TEST40')
    if any(by_id[c.case_id]['partition'] != partitions[c.case_id] for c in cases):
        raise ValueError('Export and fold0 partition assignments differ')
    return cases, partitions, preprocess, exported, by_id, paths, identity


def qc_case(case, preprocess, partition, exported, tile_stream):
    started = time.perf_counter()
    row = dict(case_id=case.case_id, partition=partition, status='FAIL', reason='')
    try:
        for path in (case.image, case.mask):
            if not path.is_file():
                raise FileNotFoundError(f'Missing source CT/GT: {path}')
        before_stat = [(p.stat().st_size, p.stat().st_mtime_ns) for p in (case.image, case.mask)]
        row.update(ct_sha256_before=file_hash(case.image), gt_sha256_before=file_hash(case.mask))
        if (row['ct_sha256_before'] != exported['ct_output_sha256']
                or row['gt_sha256_before'] != exported['gt_output_sha256']):
            raise ValueError('Source file SHA-256 differs from EXPORT_COMPLETE.json')
        row['hashes_match_export'] = True
        header = nib.load(case.image)
        if header.get_data_dtype() != np.dtype('float32'):
            raise ValueError('Precomputed CT storage must be float32; no lossy CT cast permitted')
        if header.dataobj.slope != 1.0 or header.dataobj.inter != 0.0:
            raise ValueError('Precomputed CT requires identity NIfTI scaling')
        if list(header.shape) != exported['shape']:
            raise ValueError('CT shape differs from export metadata')
        image, mask, metadata = read_arrays(case, preprocess, require_mask=True)
        if (metadata['native_size_hw'] != [512, 512] or metadata['processed_size_hw'] != [512, 512]
                or metadata['z_resampled'] or metadata['lesion_retention']['resize_applied']
                or metadata['canonical_orientation'] != ['L', 'P', 'S']):
            raise ValueError('Unexpected reader geometry/resize/resampling')
        row.update(geometry=metadata)
        row.update(qc_arrays(image, mask, metadata, case_id=case.case_id,
                             partition=partition, tile_stream=tile_stream))
        if row['gt_lesion_before'] != exported['gt_lesion_voxels']:
            raise ValueError('GT lesion count differs from export metadata')
        del image, mask
        row.update(ct_sha256_after=file_hash(case.image), gt_sha256_after=file_hash(case.mask))
        after_stat = [(p.stat().st_size, p.stat().st_mtime_ns) for p in (case.image, case.mask)]
        row['source_unchanged'] = (before_stat == after_stat
            and row['ct_sha256_before'] == row['ct_sha256_after']
            and row['gt_sha256_before'] == row['gt_sha256_after'])
        if not row['source_unchanged']:
            raise ValueError('Source CT/GT changed during read-only QC')
        row['status'] = 'PASS'
    except Exception as error:
        row['reason'] = f'{type(error).__name__}: {error}'
    finally:
        row['seconds'] = time.perf_counter() - started
        gc.collect()
    return row


def code_provenance():
    files = [Path(__file__).resolve(), *(ROOT / 'src/merit_covid/data/pipeline' / name for name in
             ('cases.py', 'config.py', 'preprocessing.py', 'geometry.py', 'slab.py', 'splits.py'))]
    result = dict(file_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in files})
    for key, command in [('git_head', ['rev-parse', 'HEAD']), ('git_branch', ['branch', '--show-current']),
                         ('git_status', ['status', '--short'])]:
        try:
            run = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT), *command],
                                 capture_output=True, text=True, timeout=10)
            result[key] = run.stdout.strip() if run.returncode == 0 else 'UNAVAILABLE_ARCHIVE_OR_NO_GIT'
        except (OSError, subprocess.TimeoutExpired):
            result[key] = 'UNAVAILABLE_ARCHIVE_OR_NO_GIT'
    return result


def write_reports(output, summary):
    with (output / 'tiling128_qc_199.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in summary['cases']:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    summary['artifact_sha256'] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
    (output / 'tiling128_qc_199.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run_qc(*, data_root, manifest='cases.json', split='splits_covid19_20_5fold.json',
           config='merit_precomputed_v3.yaml', export='EXPORT_COMPLETE.json', output_dir, case_ids=None):
    started = time.perf_counter()
    data_root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(data_root):
        raise ValueError('--output-dir must be outside the read-only input dataset')
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema_version=1, status='BLOCKED_INPUT', b2_gate_pass=False,
        expected_cases=EXPECTED_CASES, tested_cases=0, passed_cases=0, failed_cases=0,
        not_tested_cases=EXPECTED_CASES, fold=0, cases=[], failures=[],
        scope='B2 standalone CPU tiling QC of exported V3; no model integration',
        protocol=dict(tile_size=128, rows=4, cols=4, overlap=0, interpolation=False,
                      coordinate_frame='production read_arrays canonical LPS array [Z,Y,X]',
                      coordinates='zero-based, half-open; row=Y, col=X; master_z is canonical Z',
                      preprocessing='native PASS-THROUGH of already V3 CT; no second normalization',
                      slab_size=11, boundary_policy='edge', duplicate_slice_policy='unique'),
        gpu_used=False, training_updates=0, model_loaded=False, data_root=str(data_root),
        code=code_provenance())
    try:
        cases, partitions, preprocess, exported, entries, paths, identity = load_inputs(
            data_root, manifest, split, config, export)
        summary['identity'] = identity
        summary['input_metadata'] = {name: dict(path=str(path), sha256_before=file_hash(path)) for name, path in paths.items()}
        summary['export_provenance'] = {k: v for k, v in exported.items() if k != 'files'}
        summary['warnings'] = [
            'QC proves tiling preservation of this exported V3 dataset, not equivalence to original raw/compressed CT.',
            'Volume/group separation does not verify patient-level identity.',
        ]
        if exported['status'] == 'COMPLETE_WITH_PROVENANCE_LIMITATION':
            summary['warnings'].append('Exporter reports COMPLETE_WITH_PROVENANCE_LIMITATION; this QC does not remove that limitation.')
        selected = set(case_ids) if case_ids else {c.case_id for c in cases}
        if not selected <= {c.case_id for c in cases}:
            raise ValueError('Unknown --case-id: ' + ', '.join(sorted(selected - {c.case_id for c in cases})))
        selected_cases = [c for c in cases if c.case_id in selected]
        summary.update(status='RUNNING', selected_cases=len(selected_cases),
                       partition_case_counts=EXPECTED_PARTITIONS,
                       input_validation_seconds=time.perf_counter() - started)
        with gzip.open(output / 'tiling128_tile_metadata.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as stream:
            for index, case in enumerate(selected_cases, 1):
                row = qc_case(case, preprocess, partitions[case.case_id], entries[case.case_id], stream)
                summary['cases'].append(row)
                print(f'[{index}/{len(selected_cases)}] {row["status"]} {case.case_id} {row["seconds"]:.2f}s {row["reason"]}', flush=True)
        for item in summary['input_metadata'].values():
            item['sha256_after'] = file_hash(item['path'])
            item['unchanged'] = item['sha256_before'] == item['sha256_after']
        if not all(item['unchanged'] for item in summary['input_metadata'].values()):
            raise ValueError('Source manifest/split/config/export changed during QC')
        passed = [r for r in summary['cases'] if r['status'] == 'PASS']
        summary.update(tested_cases=len(selected_cases), passed_cases=len(passed),
                       failed_cases=len(selected_cases) - len(passed), not_tested_cases=EXPECTED_CASES - len(selected_cases))
        summary['failures'] = [dict(case_id=r['case_id'], reason=r['reason']) for r in summary['cases'] if r['status'] != 'PASS']
        count_keys = ('tiles_total', 'tiles_lesion', 'tiles_empty', 'gt_lesion_before', 'gt_lesion_after',
                      'voxel_pairs_checked', 'ct_mismatched_pixels', 'ct_mismatched_bits', 'gt_mismatched_pixels', 'slab_checks_passed')
        summary['partition_totals_passed_cases_only'] = {
            role: dict(cases=sum(r['partition'] == role for r in passed), **{
                key: sum(r.get(key, 0) for r in passed if r['partition'] == role) for key in count_keys})
            for role in EXPECTED_PARTITIONS}
        summary['totals_passed_cases_only'] = {key: sum(r.get(key, 0) for r in passed) for key in count_keys}
        summary['b2_gate_pass'] = len(passed) == EXPECTED_CASES and not summary['failures']
        summary['status'] = 'PASS' if summary['b2_gate_pass'] else ('FAIL' if summary['failures'] else 'PARTIAL_PASS')
    except Exception as error:
        summary['status'] = 'FAIL' if summary['cases'] else 'BLOCKED_INPUT'
        summary['failures'].append(dict(case_id=None, reason=f'{type(error).__name__}: {error}'))
        summary['tested_cases'] = len(summary['cases'])
        summary['passed_cases'] = sum(r['status'] == 'PASS' for r in summary['cases'])
        summary['failed_cases'] = summary['tested_cases'] - summary['passed_cases']
        summary['not_tested_cases'] = EXPECTED_CASES - summary['tested_cases']
        summary['b2_gate_pass'] = False
    summary['seconds_total'] = time.perf_counter() - started
    write_reports(output, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', default='cases.json')
    parser.add_argument('--split', default='splits_covid19_20_5fold.json')
    parser.add_argument('--config', default='merit_precomputed_v3.yaml')
    parser.add_argument('--export', default='EXPORT_COMPLETE.json')
    parser.add_argument('--output-dir', type=Path, required=True, help='A NEW directory outside --data-root')
    parser.add_argument('--case-id', action='append', help='Optional bounded subset; never claims 199 PASS')
    args = vars(parser.parse_args(argv))
    args['case_ids'] = args.pop('case_id')
    try:
        report = run_qc(**args)
    except (OSError, ValueError) as error:
        print(f'Cannot start QC: {error}', file=sys.stderr)
        return 2
    print(json.dumps({k: report[k] for k in ('status', 'b2_gate_pass', 'tested_cases', 'passed_cases', 'failed_cases', 'not_tested_cases', 'seconds_total', 'failures')}))
    return 0 if report['status'] in ('PASS', 'PARTIAL_PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
