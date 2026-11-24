"""Isolated B3A data-delivery prototypes, never a production model integration.

NPY volume tiles are lossless and memory-mappable. Their float64 affine and full
NIfTI spatial provenance live in one atomic export manifest; no NIfTI scaling or
qform/sform quantization is introduced by the intermediate storage format.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.geometry import canonical_shape, inspect_pair
from merit_covid.data.pipeline.preprocessing import read_arrays
from merit_covid.data.pipeline.slab import sample_arrays
from tools.qc_tiling128 import TILES, crop_tile, file_hash


@dataclass(frozen=True, order=True)
class TileSampleKey:
    original_case_id: str
    master_z: int
    tile_id: int


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def tile_geometry(full, tile_id):
    tile = TILES[tile_id]
    translation = np.eye(4)
    translation[:3, 3] = [tile.x0, tile.y0, 0]
    affine = np.asarray(full['canonical_affine_mm'], dtype=np.float64) @ translation
    return dict(canonical_shape_xyz=[128, 128, full['canonical_shape_xyz'][2]],
                canonical_affine_mm=affine.tolist(), tile_affine_mm=affine.tolist(),
                **asdict(tile), tile_id=tile_id,
                storage_order='ZYX', affine_coordinate_order='XYZ', spatial_unit='mm')


def spatial_forms(case, require_mask=True):
    result = {}
    for role, path in (('ct', case.image), ('gt', case.mask if require_mask else None)):
        if path is None:
            continue
        image = nib.load(path)
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        result[role] = dict(qform=None if qform is None else qform.tolist(), qform_code=int(qcode),
                            sform=None if sform is None else sform.tolist(), sform_code=int(scode),
                            unit=image.header.get_xyzt_units()[0])
    return result


def atomic_npy(path, array):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def export_volume_tiles(cases, config, partitions, output, *, provenance=None, expected_hashes=None):
    """Only a pilot, published atomically after every saved voxel is verified."""
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Export already exists: {output}; no overwrite. Validate/reuse its manifest or choose a NEW output.')
    if list(output.parent.glob(output.name + '.incomplete-*')):
        raise FileExistsError('Incomplete export exists; inspect it or choose a new output name. It cannot be benchmarked.')
    if not cases or len(cases) > 16:
        raise ValueError('B3A exporter is limited to 1..16 pilot original volumes, never full199')
    if config.intensity_encoding != 'native' or config.size_hw is not None:
        raise ValueError('Export requires pass-through precomputed V3, no resize')
    for case in cases:
        if output.is_relative_to(case.image.parent.resolve()):
            raise ValueError('Export must be outside the input dataset')
    stage = output.with_name(output.name + '.incomplete-' + uuid.uuid4().hex[:8])
    (stage / 'tiles').mkdir(parents=True)
    manifest = dict(schema_version=1, status='INCOMPLETE', format='npy_zyx', fold=0,
        source_provenance=provenance or {}, data_file_count=0, source_cases=[],
        sampling=dict(slab_size=11, sampling_mode='index', boundary_policy='edge', duplicate_slice_policy='unique'),
        geometry_note='NPY has no NIfTI qform/sform; source forms are retained. Tile affine is float64 full_canonical_affine_mm @ translate(x0,y0,0).')
    try:
        for case in cases:
            tick = time.perf_counter()
            source_hash = dict(ct=file_hash(case.image), gt=file_hash(case.mask))
            if expected_hashes is not None:
                expected = expected_hashes[case.case_id]
                if source_hash != dict(ct=expected['ct_output_sha256'], gt=expected['gt_output_sha256']):
                    raise ValueError(f'{case.case_id}: input hashes differ from V3 export')
            image, mask, full = read_arrays(case, config)
            if image.shape[1:] != (512, 512) or image.dtype != np.float32:
                raise ValueError('Pilot needs native512 float32')
            entry = dict(case_id=case.case_id, group_id=case.group_id, partition=partitions[case.case_id],
                source_hashes=source_hash, full_geometry=full, source_nifti_forms=spatial_forms(case), tiles=[])
            for tile_id, tile in enumerate(TILES):
                ct = np.ascontiguousarray(crop_tile(image, tile))
                gt = np.ascontiguousarray(crop_tile(mask, tile), dtype=np.uint8)
                record = dict(tile_id=tile_id, geometry=tile_geometry(full, tile_id))
                for role, array in (('ct', ct), ('gt', gt)):
                    relative = f'tiles/{case.case_id}_tile{tile_id:02d}_{role}.npy'
                    path = stage / relative
                    atomic_npy(path, array)
                    decoded = np.load(path, allow_pickle=False, mmap_mode='r')
                    if decoded.dtype != array.dtype or decoded.shape != array.shape:
                        raise ValueError('Reloaded volume tile shape/dtype changed')
                    if role == 'ct':
                        equal = np.array_equal(decoded.view(np.uint32), array.view(np.uint32))
                    else:
                        equal = np.array_equal(decoded, array) and np.isin(decoded, [0, 1]).all()
                    if not equal:
                        raise ValueError(f'Volume tile roundtrip failed: {relative}')
                    record[role] = dict(path=relative, sha256=file_hash(path), bytes=path.stat().st_size,
                                        dtype=str(array.dtype), shape=list(array.shape))
                    del decoded
                entry['tiles'].append(record)
            del image, mask, ct, gt
            if source_hash != dict(ct=file_hash(case.image), gt=file_hash(case.mask)):
                raise ValueError(f'{case.case_id}: original input changed during export')
            entry['export_seconds'] = time.perf_counter() - tick
            manifest['source_cases'].append(entry)
            manifest['data_file_count'] += 32
            print(f'EXPORT {case.case_id} {entry["export_seconds"]:.2f}s', flush=True)
            gc.collect()
        manifest.update(status='COMPLETE', export_seconds=time.perf_counter() - started,
                        data_bytes=sum(t[role]['bytes'] for c in manifest['source_cases'] for t in c['tiles'] for role in ('ct', 'gt')))
        write_json(stage / 'b3a_precompute_manifest.json', manifest)
        write_json(stage / 'EXPORT_COMPLETE.json', dict(status='COMPLETE', manifest_sha256=file_hash(stage / 'b3a_precompute_manifest.json')))
        stage.rename(output)
    except Exception as error:
        write_json(stage / 'EXPORT_ERROR.json', dict(status='INCOMPLETE', reason=f'{type(error).__name__}: {error}'))
        raise
    return output / 'b3a_precompute_manifest.json'


def validate_export(path, *, require_mask=True):
    path = Path(path).resolve()
    marker = path.parent / 'EXPORT_COMPLETE.json'
    if not marker.is_file() or json.loads(marker.read_text())['manifest_sha256'] != file_hash(path):
        raise ValueError('Incomplete export or manifest SHA mismatch; STOP')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['status'] != 'COMPLETE' or data['format'] != 'npy_zyx':
        raise ValueError('Export is not complete')
    for case in data['source_cases']:
        if [t['tile_id'] for t in case['tiles']] != list(range(16)):
            raise ValueError('Expected all16 volume tiles')
        for tile in case['tiles']:
            if tile['geometry'] != tile_geometry(case['full_geometry'], tile['tile_id']):
                raise ValueError('Tile geometry/XY translation differs')
            for role in ('ct', 'gt') if require_mask else ('ct',):
                item = tile[role]
                source = (path.parent / item['path']).resolve()
                if not source.is_relative_to(path.parent) or not source.is_file() or file_hash(source) != item['sha256']:
                    raise ValueError(f'Incomplete/corrupted tile: {source}')
    return data


class TileDeliveryDataset(Dataset):
    """B3A prototype only. ends/cases always group ORIGINAL volumes, not tiles."""

    def __init__(self, cases, config, *, mode, manifest=None, order='case_master_tile', require_mask=True):
        if mode not in ('dynamic', 'precomputed') or order not in ('case_master_tile', 'case_tile_master'):
            raise ValueError('Unknown data mode/order')
        if (config.intensity_encoding != 'native' or config.size_hw is not None
                or config.slab_size != 11 or config.boundary_policy != 'edge'
                or config.duplicate_slice_policy != 'unique' or config.sampling_mode != 'index'):
            raise ValueError('B3A requires native pass-through, no resize, 11/index/edge/unique')
        self.mode, self.order, self.config, self.require_mask = mode, order, config, require_mask
        self.cases = list(cases) if require_mask else [replace(c, mask=None) for c in cases]
        self.partition = 'train'
        self.ends = [0]
        self.export_root, self.exported = None, None
        if mode == 'precomputed':
            manifest = Path(manifest).resolve()
            marker = json.loads((manifest.parent / 'EXPORT_COMPLETE.json').read_text())
            if marker['manifest_sha256'] != file_hash(manifest):
                raise ValueError('Incomplete/corrupted manifest')
            data = json.loads(manifest.read_text(encoding='utf-8'))
            if data['status'] != 'COMPLETE':
                raise ValueError('Incomplete export')
            self.export_root = manifest.parent
            self.exported = {e['case_id']: e for e in data['source_cases']}
        self.depths = []
        for case in self.cases:
            if mode == 'dynamic':
                header, _ = inspect_pair(case, config, require_mask=require_mask)
                depth = canonical_shape(header)[2]
            else:
                entry = self.exported[case.case_id]
                if entry['group_id'] != case.group_id:
                    raise ValueError('Original group identity changed')
                depth = entry['full_geometry']['canonical_shape_xyz'][2]
            self.depths.append(depth)
            self.ends.append(self.ends[-1] + 16 * depth)
        self._cached_index = None
        self._cached = None

    def __len__(self):
        return self.ends[-1]

    def __getstate__(self):
        return {**self.__dict__, '_cached_index': None, '_cached': None}

    def key_for_index(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        case_index = bisect_right(self.ends, index) - 1
        local = index - self.ends[case_index]
        if self.order == 'case_master_tile':
            master, tile_id = divmod(local, 16)
        else:
            tile_id, master = divmod(local, self.depths[case_index])
        return TileSampleKey(self.cases[case_index].case_id, master, tile_id)

    def index_for_key(self, key):
        case_index = next(i for i, c in enumerate(self.cases) if c.case_id == key.original_case_id)
        if not 0 <= key.master_z < self.depths[case_index] or not 0 <= key.tile_id < 16:
            raise IndexError(key)
        local = key.master_z * 16 + key.tile_id if self.order == 'case_master_tile' else key.tile_id * self.depths[case_index] + key.master_z
        return self.ends[case_index] + local

    def clear_cache(self):
        self._cached = None
        self._cached_index = None

    def _load(self, case_index, tile_id):
        case = self.cases[case_index]
        cache_key = case_index if self.mode == 'dynamic' else (case_index, tile_id)
        files, file_bytes = 0, 0
        if self._cached_index != cache_key:
            self.clear_cache()
            if self.mode == 'dynamic':
                image, mask, full = read_arrays(case, self.config, require_mask=self.require_mask)
                forms = spatial_forms(case, self.require_mask)
                paths = [case.image] + ([case.mask] if self.require_mask else [])
            else:
                entry = self.exported[case.case_id]
                tile = entry['tiles'][tile_id]
                paths = [self.export_root / tile[r]['path'] for r in (('ct', 'gt') if self.require_mask else ('ct',))]
                image = np.load(paths[0], allow_pickle=False, mmap_mode='r')
                mask = np.load(paths[1], allow_pickle=False, mmap_mode='r') if self.require_mask else None
                if image.shape != (self.depths[case_index], 128, 128) or image.dtype != np.float32:
                    raise ValueError('Invalid CT volume tile shape/dtype')
                if self.require_mask and (mask.shape != image.shape or mask.dtype != np.uint8):
                    raise ValueError('Invalid GT volume tile shape/dtype')
                full = deepcopy(entry['full_geometry'])
                forms = deepcopy(entry['source_nifti_forms'])
                if not self.require_mask:
                    full.update(label_values=[], native_lesion_voxels=0)
                    forms.pop('gt', None)
            files, file_bytes = len(paths), sum(p.stat().st_size for p in paths)
            self._cached = image, mask, full, forms
            self._cached_index = cache_key
        return (*self._cached, files, file_bytes)

    def __getitem__(self, index):
        key = self.key_for_index(index)
        case_index = bisect_right(self.ends, index) - 1
        case = self.cases[case_index]
        image, mask, full, forms, files, file_bytes = self._load(case_index, key.tile_id)
        tile = TILES[key.tile_id]
        local = tile_geometry(full, key.tile_id)
        # An XY slice is only a view of the cached FULL volume: it neither
        # precomputes a tile nor copies other Z slices. Gathering the native11
        # Z indices from this view equals full-slab-then-crop bit for bit, while
        # avoiding a new512x512 slab for each of the16 tile keys.
        if self.mode == 'dynamic':
            image = crop_tile(image, tile)
            mask = crop_tile(mask, tile) if mask is not None else None
        sample = sample_arrays(image, mask, key.master_z, 11, local,
                               sampling_mode='index', boundary_policy='edge', duplicate_slice_policy='unique')
        metadata = dict(full_geometry=full, tile_geometry=local, source_nifti_forms=forms,
            original_affine_mm=full['original_affine_mm'], original_shape_xyz=full['original_shape_xyz'],
            full_volume_canonical_affine_mm=full['canonical_affine_mm'],
            full_volume_canonical_shape_xyz=full['canonical_shape_xyz'],
            tile_affine_mm=local['tile_affine_mm'])
        return {**{name: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for name, value in sample.items()},
                'case_id': case.case_id, 'group_id': case.group_id,
                'master_z': key.master_z, 'tile_id': key.tile_id, **asdict(tile),
                'metadata': metadata, 'delivery_opened_files': files,
                'delivery_declared_file_bytes': file_bytes}


class FixedItinerary(Sampler):
    """Explicit sample indices; cursor counts CONSUMED samples, not prefetch draws."""
    def __init__(self, indices, *, start=0):
        self.indices = tuple(indices)
        if not 0 <= start <= len(self.indices):
            raise ValueError('Invalid consumed-sample cursor')
        self.start = start

    def __iter__(self):
        return iter(self.indices[self.start:])

    def __len__(self):
        return len(self.indices) - self.start


def compare_samples(a, b):
    exact, float_max = 0, 0.0
    for key in a:
        if key.startswith('delivery_'):
            continue
        left, right = a[key], b[key]
        if isinstance(left, torch.Tensor):
            if left.dtype != right.dtype or left.shape != right.shape:
                raise AssertionError(f'{key}: dtype/shape mismatch')
            if key in ('image_slab', 'mask_slab', 'master_mask'):
                if not torch.equal(left.view(torch.int32), right.view(torch.int32)):
                    raise AssertionError(f'{key}: bitwise mismatch')
                exact += left.numel()
            elif key in ('slice_positions_mm', 'z_offsets_mm', 'requested_offsets_mm'):
                error = float((left - right).abs().max())
                float_max = max(float_max, error)
                if error > 1e-6:
                    raise AssertionError(f'{key}: physical coordinate mismatch {error}')
            elif not torch.equal(left, right):
                raise AssertionError(f'{key}: tensor mismatch')
        elif left != right:
            raise AssertionError(f'{key}: metadata/value mismatch')
    if set(a) != set(b):
        raise AssertionError('Sample field sets differ')
    return exact, float_max
