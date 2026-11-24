"""TRAIN-only fitting for the shared CT reader; no GT-based selection or Torch.

Padding is a conservative, explicit fit-domain heuristic, not verified anatomy.
Only stable boundary-connected -2048 is excluded; inference needs no mask.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import label, generate_binary_structure

POLICY = 'train_fovaware_clipped_zscore_v1'
RULE = dict(version=1, sentinel=-2048, connectivity=4, min_slices=3,
            min_area_fraction=.05, max_area_fraction=.30,
            central_guard_fraction=.50, max_edge_depth_fraction=.20,
            stability='bitwise identical boundary-connected mask on every slice',
            corner_policy='all four corners; every excluded component touches a corner',
            unknown_fallback='all finite voxels; no other sentinel is inferred',
            max_fallback_case_fraction=.20)
QUANTUM = 1 / 16  # NIfTI-scaled intensity units, NOT a HU claim.


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _boundary_connected(plane):
    labels, _ = label(plane, structure=generate_binary_structure(2, 1))
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    border = border[border != 0]
    return np.isin(labels, border), labels


def identify_fit_padding(image_xyz):
    """Return a stable XY exclusion mask or None (fallback includes ALL voxels).

    No circle, GT, erosion/dilation, or changes to input values are used. Noncircle
    corner/edge patterns can pass the same outer-band, stability and area checks.
    A connected band reaching the central half or >20% depth is ambiguous.
    """
    if image_xyz.ndim != 3 or min(image_xyz.shape) < 1 or not np.isfinite(image_xyz).all():
        raise ValueError('Padding fit requires finite nonempty scaled CT XYZ')
    nx, ny, nz = image_xyz.shape
    total = int(image_xyz.size)
    row = dict(status='UNKNOWN', reason='', unverified_background_included=True,
               total_voxels=total, excluded_voxels=0, sentinel_voxels=0,
               boundary_components=0, stable_boundary=False)
    reference = image_xyz[:, :, 0] == RULE['sentinel']
    boundary, labels = _boundary_connected(reference)
    sentinel_count = int(reference.sum())
    stable = True
    for z in range(1, nz):
        sentinel = image_xyz[:, :, z] == RULE['sentinel']
        sentinel_count += int(sentinel.sum())
        if not np.array_equal(sentinel, reference):
            current, _ = _boundary_connected(sentinel)
            stable = stable and np.array_equal(boundary, current)
    row['sentinel_voxels'] = sentinel_count
    row['stable_boundary'] = stable
    if not sentinel_count:
        row['reason'] = 'no_minus2048; other encodings (including -1024) are not inferred'
        return None, row
    row['status'] = 'AMBIGUOUS'
    fraction = float(boundary.mean())
    row['boundary_area_fraction'] = fraction
    component_ids = set(np.unique(labels[boundary])) - {0}
    corner_ids = {int(labels[x, y]) for x, y in ((0, 0), (0, ny-1), (nx-1, 0), (nx-1, ny-1))}
    row['boundary_components'] = len(component_ids)
    row['isolated_sentinel_voxels_kept'] = sentinel_count - int(boundary.sum()) * nz if stable else None
    xx, yy = np.nonzero(boundary)
    depth = int(np.minimum.reduce([xx, nx-1-xx, yy, ny-1-yy]).max()) if len(xx) else 0
    row['max_edge_depth_pixels'] = depth
    guard = boundary[nx//4:nx-nx//4, ny//4:ny-ny//4]
    if nz < RULE['min_slices']:
        row['reason'] = 'fewer_than_3_slices_for_stability'
    elif not stable:
        row['reason'] = 'boundary_component_changes_between_slices'
    elif not component_ids or 0 in corner_ids or component_ids != corner_ids:
        row['reason'] = 'missing_corner_anchor_or_additional_boundary_component'
    elif not RULE['min_area_fraction'] <= fraction <= RULE['max_area_fraction']:
        row['reason'] = 'excluded_area_outside_predeclared_5_to_30_percent'
    elif guard.any() or depth > RULE['max_edge_depth_fraction'] * min(nx, ny):
        row['reason'] = 'boundary_sentinel_reaches_central_guard_or_too_deep'
    else:
        row.update(status='CONFIDENT', reason='stable_corner_anchored_outer_band',
                   excluded_voxels=int(boundary.sum()) * nz, unverified_background_included=False)
        return boundary, row
    return None, row


class QuantizedHistogram:
    """Deterministic pooled quantile approximation, absolute error <=1/32 unit.

    Round each scaled float32 voxel to nearest multiple of 1/16 (ties to even),
    count bins exactly, then apply numpy linear-quantile rank interpolation.
    Integer intensities are represented exactly. No random sample/seed is used.
    """
    def __init__(self):
        self.bins = Counter()
        self.count = 0

    def add(self, values):
        if not values.size or not np.isfinite(values).all():
            raise ValueError('Empty/nonfinite quantile block')
        scaled = values.astype(np.float64) / QUANTUM
        if np.max(np.abs(scaled)) >= 2**62:
            raise ValueError('Intensity outside safe quantization domain; fit BLOCKED')
        q = np.rint(scaled).astype(np.int64)
        lo, hi = int(q.min()), int(q.max())
        if hi-lo <= 2_000_000:
            counts = np.bincount(q-lo)
            indices = np.flatnonzero(counts)
            self.bins.update({int(i)+lo:int(counts[i]) for i in indices})
        else:
            indices, counts = np.unique(q, return_counts=True)
            self.bins.update({int(i):int(n) for i, n in zip(indices, counts)})
        self.count += values.size
        if len(self.bins) > 2_000_000:
            raise ValueError('Too many histogram bins; fit BLOCKED to bound RAM')

    def percentile(self, percentile):
        if not self.count:
            raise ValueError('No accepted TRAIN voxels')
        keys = sorted(self.bins)
        cumulative = np.cumsum([self.bins[k] for k in keys], dtype=np.int64)
        rank = (self.count-1)*percentile/100
        a, b = int(np.floor(rank)), int(np.ceil(rank))
        va = keys[int(np.searchsorted(cumulative, a, side='right'))]*QUANTUM
        vb = keys[int(np.searchsorted(cumulative, b, side='right'))]*QUANTUM
        return float(va+(vb-va)*(rank-a))


def accepted_blocks(array, padding, block_depth=4):
    for z in range(0, array.shape[2], block_depth):
        block = array[:, :, z:z+block_depth]
        # Only XY voxels in the stable padding mask are omitted, on every Z.
        yield block[~padding].reshape(-1) if padding is not None else block.reshape(-1)


def clipped_moments(blocks, lower, upper):
    count, mean, m2 = 0, 0., 0.
    for values in blocks:
        clipped = np.clip(values.astype(np.float64), lower, upper)
        n = int(clipped.size)
        if not n:
            raise ValueError('Empty accepted block')
        mu = float(clipped.mean())
        variance = float(clipped.var())
        total, delta = count+n, mu-mean
        m2 += variance*n+delta*delta*count*n/total
        mean += delta*n/total
        count = total
    return count, mean, m2


def combine_moments(left, right):
    n, mean, m2 = left
    k, mu, other = right
    total, delta = n+k, mu-mean
    return total, mean+delta*k/total, m2+other+delta*delta*n*k/total


def validate_fit_stats(stats):
    """No filesystem/GT access: sufficient for frozen CT-only inference."""
    if not isinstance(stats, dict) or stats.get('status') != 'FITTED':
        raise ValueError('V3 stats NOT_FITTED/BLOCKED or missing; run the CPU fit first')
    if stats.get('schema_version') != 1 or stats.get('policy') != POLICY:
        raise ValueError('V3 stats schema/policy mismatch')
    if stats.get('stats_sha256') != digest({k:v for k,v in stats.items() if k != 'stats_sha256'}):
        raise ValueError('V3 fit stats checksum mismatch')
    p = stats.get('parameters', {})
    if set(p) != {'lower', 'upper', 'mean', 'std'} or not all(type(v) in (int,float) and np.isfinite(v) for v in p.values()):
        raise ValueError('V3 requires finite L/U/mean/std')
    if p['lower'] >= p['upper'] or p['std'] <= 1e-8:
        raise ValueError('V3 requires L<U and std>1e-8')
    ids = stats.get('case_ids')
    if (stats.get('fold') != 0 or not isinstance(ids, list) or len(ids) != 127
            or ids != sorted(set(ids)) or stats.get('case_ids_sha256') != digest(ids)):
        raise ValueError('V3 requires exactly the 127 unique fold0 TRAIN IDs')
    for key in ('split_sha256_lf', 'manifest_sha256', 'manifest_fingerprint'):
        if not isinstance(stats.get(key), str) or len(stats[key]) != 64 or any(c not in '0123456789abcdef' for c in stats[key]):
            raise ValueError('V3 missing/invalid provenance: '+key)
    if stats.get('padding_rule') != RULE:
        raise ValueError('V3 padding rule differs from the implemented policy')
    if stats.get('voxel_counts', {}).get('accepted', 0) < 2:
        raise ValueError('No accepted TRAIN population')
    if stats.get('source_ct_sha256', {}).keys() != set(ids):
        raise ValueError('V3 source CT hashes must cover exactly TRAIN')
    return stats


def validate_fit_binding(stats, manifest, split, *, fold=0, cases=None):
    """Check actual saved split/manifest, never trust a self-declared case list."""
    validate_fit_stats(stats)
    manifest, split_path = Path(manifest), Path(split)
    raw = split_path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    selected = next(f for f in data['folds'] if f['fold'] == fold)['train']['case_ids']
    if (fold != stats['fold'] or set(selected) != set(stats['case_ids'])
            or hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() != stats['split_sha256_lf']
            or file_hash(manifest) != stats['manifest_sha256']
            or data['manifest_fingerprint'] != stats['manifest_fingerprint']):
        raise ValueError('V3 stats do not match the manifest/fold/split TRAIN partition')
    if cases is not None and {c.case_id for c in cases} != set(stats['case_ids']):
        raise ValueError('V3 fitting must receive exactly TRAIN, no VAL/TEST IDs')


def fit_population(cases, read_ct, *, block_depth=4):
    """Two passes over one bounded CT at a time; returns numerical result/audit.

    read_ct(case) is CT-only. The canonical driver validates 127 TRAIN identities
    and native512 before using this core; small arrays can exercise it in tests.
    """
    histogram, rows, masks = QuantizedHistogram(), [], {}
    for i, case in enumerate(cases, 1):
        image = read_ct(case)
        padding, row = identify_fit_padding(image)
        row['case_id'] = case.case_id
        row['accepted_voxels'] = row['total_voxels']-row['excluded_voxels']
        rows.append(row)
        masks[case.case_id] = None if padding is None else (padding.shape, np.packbits(padding))
        for block in accepted_blocks(image, padding, block_depth):
            histogram.add(block)
        print('FIT histogram',i,len(cases),case.case_id,row['status'],flush=True)
        del image, padding
    fallback = sum(row['status'] != 'CONFIDENT' for row in rows)
    if fallback/len(rows) > RULE['max_fallback_case_fraction']:
        return None, rows, 'More than 20% cases require unverified fallback; review padding policy'
    lower, upper = histogram.percentile(.1), histogram.percentile(99.9)
    if lower >= upper:
        return None, rows, 'Degenerate percentile bounds'
    moments = (0, 0., 0.)
    for i, case in enumerate(cases, 1):
        image = read_ct(case)
        stored = masks[case.case_id]
        padding = None if stored is None else np.unpackbits(stored[1], count=int(np.prod(stored[0]))).reshape(stored[0]).astype(bool)
        moment = clipped_moments(accepted_blocks(image, padding, block_depth), lower, upper)
        if moment[0] != rows[i-1]['accepted_voxels']:
            raise ValueError('Fit accounting changed between passes')
        moments = combine_moments(moments, moment)
        print('FIT clipped moments',i,len(cases),case.case_id,flush=True)
        del image, padding
    count, mean, m2 = moments
    std = float(np.sqrt(m2/count))
    if count != histogram.count or not np.isfinite([lower, upper, mean, std]).all() or std <= 1e-8:
        return None, rows, 'Invalid clipped population moments/accounting'
    return dict(lower=lower, upper=upper, mean=mean, std=std), rows, None


def fit_saved_fold(config, *, fold=0, block_depth=4, max_volume_mib=768):
    """No CT/GT writes and no VAL/TEST/GT voxel reads; no GPU/model calls."""
    from .cases import load_cases
    from .splits import validate_splits
    from .config import PreprocessConfig
    from .geometry import inspect_pair, canonical_shape, _lps_transform
    if type(fold) is not int or fold != 0 or type(block_depth) is not int or block_depth < 1:
        raise ValueError('V3 currently fits fold0 only; block_depth must be positive')
    if config['preprocess']['intensity_encoding'] != 'native':
        raise ValueError('Fit V3 from the native source config, never from transformed voxels')
    started = time.monotonic()
    data = config['data']
    cases, identity = load_cases(data['manifest'], patient_map=data.get('patient_map'))
    raw = Path(data['split_path']).read_bytes()
    original_manifest_hash = file_hash(data['manifest'])
    split = json.loads(raw.decode('utf-8-sig'))
    validate_splits(split, cases)
    part = next(f for f in split['folds'] if f['fold'] == fold)
    ids = sorted(part['train']['case_ids'])
    if len(ids) != 127 or len(set(ids)) != 127:
        raise ValueError('V3 requires the saved 127 unique TRAIN cases')
    selected = [c for c in cases if c.case_id in set(ids)]
    if {c.case_id for c in selected} != set(ids):
        raise ValueError('Missing or unexpected TRAIN case')
    native = PreprocessConfig(assume_mm=config['preprocess']['assume_mm'])
    before = {c.case_id:file_hash(c.image) for c in selected}

    def read(case):
        image, _ = inspect_pair(case, native, require_mask=False)
        if canonical_shape(image)[:2] != (512,512):
            raise ValueError(case.case_id+': V3 requires native512; no resize')
        if np.prod(image.shape)*4 > max_volume_mib*1024**2:
            raise ValueError(case.case_id+': one scaled CT exceeds CPU RAM cap')
        # One gzip decode per volume/pass. Never concatenate the TRAIN cohort.
        array = image.get_fdata(dtype=np.float32, caching='unchanged')
        return nib.orientations.apply_orientation(array, _lps_transform(image))

    parameters, rows, blocked = fit_population(selected, read, block_depth=block_depth)
    after = {c.case_id:file_hash(c.image) for c in selected}
    if before != after:
        raise ValueError('TRAIN source CT bytes changed during fit; refusing stats')
    if file_hash(data['manifest']) != original_manifest_hash or Path(data['split_path']).read_bytes() != raw:
        raise ValueError('Manifest/split changed during fit; refusing stats')
    stats = dict(schema_version=1, status='BLOCKED' if blocked else 'FITTED', policy=POLICY,
        parameters=parameters, fold=0, case_ids=ids, case_ids_sha256=digest(ids),
        split_sha256_lf=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest(),
        manifest_sha256=file_hash(data['manifest']), manifest_fingerprint=split['manifest_fingerprint'],
        source_ct_sha256=before, source_bytes_unchanged=True,
        source_values='NIfTI-scaled float32, identical to shared reader; HU unverified',
        identity=identity, independence='volume/group-disjoint; patient identity unverified',
        padding_rule=RULE, padding_counts=dict(Counter(row['status'] for row in rows)),
        voxel_counts={key:sum(row[key+'_voxels'] for row in rows) for key in ('total','accepted','excluded','sentinel')},
        quantiles=dict(percentiles=[.1,99.9], algorithm='pooled deterministic nearest-1/16 histogram; linear ranks',
                       resolution=QUANTUM, absolute_error_bound=QUANTUM/2, seed=None),
        mean_std='population moments of actual accepted scaled float32 voxels AFTER clipping; float64 Chan/Welford',
        fit_code_sha256=file_hash(__file__), elapsed_seconds=time.monotonic()-started,
        memory_policy=dict(one_scaled_ct_limit_mib=max_volume_mib, block_depth=block_depth, packed_XY_masks=True),
        reason=blocked, read_scope='exactly 127 TRAIN CT; no GT or held-out headers/voxels used to fit')
    stats['stats_sha256'] = digest(stats)
    if not blocked:
        validate_fit_binding(stats, data['manifest'], data['split_path'], cases=selected)
    return stats, rows
