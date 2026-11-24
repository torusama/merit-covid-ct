"""B3A CPU correctness and full DataLoader comparison; no model or training."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import nibabel as nib
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.dataloader import collate_slabs, seed_worker
from tools.b3a_delivery import (TileDeliveryDataset, TileSampleKey, FixedItinerary,
    export_volume_tiles, validate_export, compare_samples, write_json)
from tools.qc_tiling128 import TILES, file_hash, load_inputs, stitch_tiles


def select_pilot(cases, entries, partitions):
    """A0003 plus12 TRAIN cases: deterministic extremes plus seed42 fill."""
    anchor = next(c for c in cases if c.case_id == 'volume-covid19-A-0003')
    pool = [c for c in cases if partitions[c.case_id] == 'train' and c != anchor]
    selected = {anchor.case_id}
    criteria = [lambda c: entries[c.case_id]['shape'][2],
                lambda c: entries[c.case_id]['gt_lesion_voxels'] / np.prod(entries[c.case_id]['shape'])]
    # Read header spacing only: no new labels or normalization fitting.
    spacing = {c.case_id: float(nib.load(c.image).header.get_zooms()[2]) for c in pool}
    criteria.append(lambda c: spacing[c.case_id])
    for criterion in criteria:
        ranked = sorted(pool, key=lambda c: (criterion(c), c.case_id))
        selected.update([ranked[0].case_id, ranked[-1].case_id])
    rng = np.random.default_rng(42)
    remaining = sorted(c.case_id for c in pool if c.case_id not in selected)
    for name in rng.permutation(remaining):
        if len(selected) == 13:
            break
        selected.add(str(name))
    # Fixed case order for both modes and both layouts, derived from seed42.
    ordered = [next(c for c in cases if c.case_id == name) for name in sorted(selected)]
    permutation = np.random.default_rng(42).permutation(len(ordered))
    return [ordered[i] for i in permutation]


def source_versions():
    names = ['tools/b3a_delivery.py', 'tools/b3a_compare.py', 'tools/qc_tiling128.py',
        'tests/test_b3a_delivery.py', *['src/merit_covid/data/pipeline/' + name + '.py' for name in
        ('dataset', 'dataloader', 'slab', 'preprocessing', 'config', 'geometry', 'cases', 'splits')],
        'src/merit_covid/engine/slab_inference.py']
    result = dict(files={name: file_hash(ROOT / name) for name in names})
    for key, args in [('head', ['rev-parse', 'HEAD']), ('branch', ['branch', '--show-current']), ('status', ['status', '--short'])]:
        run = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT), *args], capture_output=True, text=True)
        result[key] = run.stdout.strip() if run.returncode == 0 else 'SOURCE_ARCHIVE_NO_GIT'
    return result


def environment():
    return dict(os=platform.platform(), cpu=platform.processor(), ram_total=psutil.virtual_memory().total,
        ram_available=psutil.virtual_memory().available, python=sys.version, numpy=np.__version__,
        nibabel=nib.__version__, torch=torch.__version__, psutil=psutil.__version__,
        cpu_logical=psutil.cpu_count(), cpu_physical=psutil.cpu_count(logical=False), torch_threads=1,
        gpu_used=False, model_loaded=False, training_updates=0)


def correctness_a0003(cases, config, manifest, output, xml_path):
    tick = time.perf_counter()
    suites = ET.parse(xml_path).getroot()
    suite_list = [suites] if suites.tag == 'testsuite' else list(suites.iter('testsuite'))
    tests = sum(int(s.attrib.get('tests', 0)) for s in suite_list)
    errors = sum(int(s.attrib.get('errors', 0)) + int(s.attrib.get('failures', 0)) + int(s.attrib.get('skipped', 0)) for s in suite_list)
    result = dict(status='RUNNING', synthetic_tests=tests, synthetic_failures_errors_or_skips=errors,
                  synthetic_xml_sha256=file_hash(xml_path), real_case='volume-covid19-A-0003',
                  matched_samples=0, exact_tensor_values=0, coordinate_max_abs_error_mm=0.,
                  real_geometry_and_full_restore=False, model_test=False)
    write_json(output, result)
    if tests < 16 or errors:
        result.update(status='FAIL', reason='Required synthetic gate failed/incomplete')
        write_json(output, result)
        raise ValueError(result['reason'])
    case = next(c for c in cases if c.case_id == result['real_case'])
    a = TileDeliveryDataset([case], config, mode='dynamic', order='case_tile_master')
    b = TileDeliveryDataset([case], config, mode='precomputed', manifest=manifest, order='case_tile_master')
    try:
        if a.depths != [291] or len(a) != len(b) or len(a) != 4656:
            raise ValueError('A0003 expected291 slices and4656 tile masters')
        for index in range(len(a)):
            left, right = a[index], b[index]
            count, error = compare_samples(left, right)
            result['matched_samples'] += 1
            result['exact_tensor_values'] += count
            result['coordinate_max_abs_error_mm'] = max(result['coordinate_max_abs_error_mm'], error)
            if index % 512 == 0:
                print(f'CORRECTNESS {index}/{len(a)}', flush=True)
        data = validate_export(manifest)
        entry = next(c for c in data['source_cases'] if c['case_id'] == case.case_id)
        image, mask, full, _, _, _ = a._load(0, 0)
        restored_ct = stitch_tiles((TILES[t['tile_id']], np.load(manifest.parent / t['ct']['path'], mmap_mode='r')) for t in entry['tiles'])
        restored_gt = stitch_tiles((TILES[t['tile_id']], np.load(manifest.parent / t['gt']['path'], mmap_mode='r')) for t in entry['tiles'])
        ct_mismatches = int(np.count_nonzero(restored_ct.view(np.uint32) != image.view(np.uint32)))
        gt_mismatches = int(np.count_nonzero(restored_gt != mask))
        if ct_mismatches or gt_mismatches or np.count_nonzero(restored_gt) != 738319:
            raise ValueError('A0003 full reconstruction/lesion count mismatch')
        from merit_covid.data.pipeline.geometry import restore_volume
        original_gt = restore_volume(restored_gt, full)
        if not np.array_equal(original_gt, np.asanyarray(nib.load(case.mask).dataobj)):
            raise ValueError('A0003 GT restore differs from native source')
        result.update(status='PASS', depth=291, ct_mismatches=ct_mismatches, gt_mismatches=gt_mismatches,
            lesion_voxels_before=int(np.count_nonzero(mask)), lesion_voxels_after=int(np.count_nonzero(restored_gt)),
            edge_masters_verified=[0, 1, 5, 145, 285, 290], real_geometry_and_full_restore=True,
            seconds=time.perf_counter() - tick, export_hashes_verified=True)
        del image, mask, restored_ct, restored_gt, original_gt
    except Exception as error:
        result.update(status='FAIL', reason=f'{type(error).__name__}: {error}', seconds=time.perf_counter() - tick)
        write_json(output, result)
        raise
    finally:
        a.clear_cache(); b.clear_cache(); gc.collect()
    write_json(output, result)
    return result


class ProcessMonitor:
    """Sample main+descendant RSS; keep departed worker I/O observations."""
    def __init__(self):
        self.main = psutil.Process()
        self.stop_event = threading.Event()
        self.peak = 0
        self.max_processes = 0
        self.io_initial, self.io_last = {}, {}
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        try:
            processes = [self.main, *self.main.children(recursive=True)]
        except psutil.Error:
            processes = [self.main]
        rss = 0
        for process in processes:
            try:
                rss += process.memory_info().rss
                counters = process.io_counters()
                values = (counters.read_bytes, counters.write_bytes, counters.read_count)
                if process.pid not in self.io_initial:
                    self.io_initial[process.pid] = values if process.pid == self.main.pid else (0, 0, 0)
                self.io_last[process.pid] = values
            except psutil.Error:
                pass
        self.peak = max(self.peak, rss)
        self.max_processes = max(self.max_processes, len(processes))

    def loop(self):
        while not self.stop_event.wait(.02):
            self.sample()

    def __enter__(self):
        self.sample(); self.thread.start(); return self

    def __exit__(self, *args):
        self.sample(); self.stop_event.set(); self.thread.join()

    def result(self):
        totals = [0, 0, 0]
        for pid, values in self.io_last.items():
            initial = self.io_initial[pid]
            for index in range(3):
                totals[index] += max(0, values[index] - initial[index])
        return dict(rss_peak_main_plus_workers_bytes=self.peak, max_processes_observed=self.max_processes,
                    process_io_read_bytes_sampled=totals[0], process_io_write_bytes_sampled=totals[1],
                    process_io_read_count_sampled=totals[2])


def run_pass(dataset, indices, workers, batch_size, name):
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=FixedItinerary(indices),
        num_workers=workers, collate_fn=collate_slabs, worker_init_fn=seed_worker,
        generator=generator, pin_memory=False, persistent_workers=False,
        **({'prefetch_factor': 2} if workers else {}))
    started = time.perf_counter()
    first = None
    checksum = 0.
    observed_keys, source_opens, declared_bytes, batches, ct_bytes, gt_bytes = [], 0, 0, 0, 0, 0
    dataset.clear_cache()
    with ProcessMonitor() as monitor:
        for batch in loader:
            # Force the full CPU batch tensors, not a reshape/header-only timing.
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and not key.startswith('delivery_'):
                    checksum += float(value.double().sum())
            if first is None:
                first = time.perf_counter() - started
            batches += 1
            ct_bytes += batch['image_slab'].numel() * batch['image_slab'].element_size()
            gt_bytes += sum(batch[k].numel() * batch[k].element_size() for k in ('mask_slab', 'master_mask'))
            source_opens += int(batch['delivery_opened_files'].sum())
            declared_bytes += int(batch['delivery_declared_file_bytes'].sum())
            for i, case in enumerate(batch['case_id']):
                observed_keys.append(TileSampleKey(case, int(batch['master_z'][i]), int(batch['tile_id'][i])))
            if batches % 2000 == 0:
                print(f'PASS {name} batches={batches} seconds={time.perf_counter()-started:.1f}', flush=True)
    seconds = time.perf_counter() - started
    expected = [dataset.key_for_index(i) for i in indices]
    if observed_keys != expected:
        raise AssertionError('DataLoader sample order/coverage differs from the canonical itinerary')
    count = len(observed_keys)
    result = dict(status='PASS', seconds=seconds, first_batch_seconds=first, batches=batches,
        samples=count, tiles_per_second=count / seconds, original_masters_equivalent_per_second=count / seconds / 16,
        ct_payload_bytes=ct_bytes, gt_payload_bytes=gt_bytes, declared_volume_open_files=source_opens,
        declared_open_file_bytes=declared_bytes, consumer_checksum=checksum,
        itinerary_sha256=hashlib.sha256(json.dumps([asdict(k) for k in observed_keys], separators=(',', ':')).encode()).hexdigest(),
        **monitor.result())
    dataset.clear_cache(); gc.collect()
    return result


def benchmark(cases, config, manifest, work, workers_list, batch_size, repetitions):
    if json.loads((work / 'b3a_correctness.json').read_text())['status'] != 'PASS':
        raise ValueError('Correctness STOP gate is not PASS')
    data = json.loads(manifest.read_text())
    report = dict(status='RUNNING', environment=environment(), source=source_versions(),
        kaggle_status='RUNNING' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else 'KAGGLE_BENCHMARK_NOT_RUN',
        cache_policy='one original full volume per worker A; one memory-mapped volume tile per worker B',
        cache_temperature='uncontrolled OS cache, warmup before measurements; NOT controlled cold-cache',
        rss_note='20ms sampled sum of main+descendant RSS; shared pages can be counted more than once; not unique physical RAM',
        io_note='OS process I/O sampled, includes logical/IPC operations; declared file bytes are full file sizes on cache misses, NOT actual mapped pages read',
        consumer='no-op CPU tensor materialization/sums, no model; workers startup and teardown included',
        batch_size=batch_size, pin_memory=False, prefetch_factor_workers=2, persistent_workers=False,
        seed=42, repetitions=repetitions, scope='ALL tile-master keys of the same13 original TRAIN pilot volumes',
        warmups=[], raw=[], skipped=[], aggregates=[],
        export_seconds=data['export_seconds'], export_data_bytes=data['data_bytes'], export_data_files=data['data_file_count'])
    write_json(work / 'b3a_benchmark.json', report)
    for order in ('case_master_tile', 'case_tile_master'):
        for workers in workers_list:
            # Conservative peak estimate for simultaneous production read_arrays
            # float64 GT validation + casts + canonical contiguous copies.
            largest = max(c['full_geometry']['canonical_shape_xyz'][2] for c in data['source_cases'])
            required = workers * largest * 512 * 512 * 20 + 768 * 2**20
            available = psutil.virtual_memory().available
            if workers and available < required:
                report['skipped'].append(dict(order=order, workers=workers, status='NOT_RUN_MEMORY_GATE',
                    available_bytes=available, required_headroom_bytes=required,
                    reason='Both modes skipped to keep comparison paired; insufficient headroom for concurrent full-volume readers'))
                write_json(work / 'b3a_benchmark.json', report)
                continue
            datasets = {mode: TileDeliveryDataset(cases, config, mode=mode, manifest=manifest, order=order)
                        for mode in ('dynamic', 'precomputed')}
            # Same indices/order in BOTH modes, covering every tile and master.
            indices = list(range(len(datasets['dynamic'])))
            assert len(datasets['precomputed']) == len(indices)
            for mode in ('dynamic', 'precomputed'):
                print(f'WARMUP {mode} {order} workers={workers} samples={len(indices)}', flush=True)
                warm = run_pass(datasets[mode], indices, workers, batch_size, f'warmup-{mode}-{order}-w{workers}')
                report['warmups'].append(dict(mode=mode, order=order, workers=workers, **warm))
                write_json(work / 'b3a_benchmark.json', report)
            for repetition in range(repetitions):
                modes = ('dynamic', 'precomputed') if repetition % 2 == 0 else ('precomputed', 'dynamic')
                for mode in modes:
                    print(f'MEASURE {repetition+1}/{repetitions} {mode} {order} workers={workers}', flush=True)
                    row = dict(mode=mode, order=order, workers=workers, repetition=repetition,
                        **run_pass(datasets[mode], indices, workers, batch_size, f'{mode}-{order}-w{workers}-r{repetition}'))
                    report['raw'].append(row)
                    print(json.dumps(row), flush=True)
                    write_json(work / 'b3a_benchmark.json', report)
            del datasets
            gc.collect()
    import csv
    if report['raw']:
        with (work / 'b3a_benchmark.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report['raw'][0]))
            writer.writeheader(); writer.writerows(report['raw'])
    for order in ('case_master_tile', 'case_tile_master'):
        for workers in workers_list:
            pair = {}
            for mode in ('dynamic', 'precomputed'):
                rows = [r for r in report['raw'] if r['mode'] == mode and r['order'] == order and r['workers'] == workers]
                if len(rows) != repetitions:
                    continue
                values = [r['seconds'] for r in rows]
                pair[mode] = dict(median_seconds=statistics.median(values),
                    min_seconds=min(values), max_seconds=max(values), stdev_seconds=statistics.stdev(values),
                    median_tiles_per_second=statistics.median(r['tiles_per_second'] for r in rows),
                    median_first_batch_seconds=statistics.median(r['first_batch_seconds'] for r in rows),
                    peak_rss_bytes=max(r['rss_peak_main_plus_workers_bytes'] for r in rows),
                    samples=rows[0]['samples'])
            if len(pair) == 2:
                gain = pair['dynamic']['median_seconds'] - pair['precomputed']['median_seconds']
                report['aggregates'].append(dict(order=order, workers=workers, **pair,
                    throughput_gain_percent=(pair['precomputed']['median_tiles_per_second'] / pair['dynamic']['median_tiles_per_second'] - 1) * 100,
                    pilot_epoch_break_even=math.ceil(data['export_seconds'] / gain) if gain > 0 else None,
                    break_even_scope='measured full13-case pilot epochs, excludes staging/upload; NOT full127/199 epochs'))
    report['status'] = 'PASS_WITH_RESOURCE_LIMIT' if report['skipped'] else 'PASS'
    if report['kaggle_status'] == 'RUNNING':
        report['kaggle_status'] = 'KAGGLE_BENCHMARK_' + report['status']
    write_json(work / 'b3a_benchmark.json', report)
    return report


def prepare(data_root, work):
    cases, parts, config, exported, entries, paths, identity = load_inputs(
        data_root, 'cases.json', 'splits_covid19_20_5fold.json', 'merit_precomputed_v3.yaml', 'EXPORT_COMPLETE.json')
    chosen = select_pilot(cases, entries, parts)
    if len(chosen) != 13 or any(parts[c.case_id] != 'train' for c in chosen):
        raise ValueError('Pilot must be A0003 plus12 original TRAIN volumes')
    estimated_tile_bytes = sum(entries[c.case_id]['shape'][2] * 512 * 512 * 5 for c in chosen)
    required = estimated_tile_bytes * 3 + 512 * 2**20
    if shutil.disk_usage(work).free < required:
        raise OSError(f'Need at least {required} free bytes for paired staging/export')
    target = work / 'storage/source_v3'
    target.mkdir(parents=True)
    tick = time.perf_counter(); staged = []; records = []
    original_hashes = {name: file_hash(path) for name, path in paths.items()}
    for case in chosen:
        destinations = []
        for role, path in (('ct', case.image), ('gt', case.mask)):
            destination = target / path.name
            shutil.copy2(path, destination)
            expected = entries[case.case_id]['ct_output_sha256' if role == 'ct' else 'gt_output_sha256']
            if file_hash(destination) != expected or file_hash(path) != expected:
                raise ValueError(f'{case.case_id}: staged source hash mismatch')
            destinations.append(destination)
        staged.append(replace(case, image=destinations[0], mask=destinations[1]))
        records.append(dict(case_id=case.case_id, group_id=case.group_id, partition=parts[case.case_id],
            shape=entries[case.case_id]['shape'], spacing_mm=list(nib.load(case.image).header.get_zooms()[:3]),
            lesion_voxels=entries[case.case_id]['gt_lesion_voxels'],
            source_paths=[str(case.image), str(case.mask)], staged_paths=list(map(str, destinations))))
    pilot = dict(seed=42, case_ids=[c.case_id for c in staged], cases=records,
        original_case_count=13, full_tile_master_count=sum(e['shape'][2] * 16 for e in records),
        selection='TRAIN-only depth/spacing/lesion-fraction extremes plus seed42 fill; benchmark uses ALL masters/tiles, no balancing',
        staging_seconds=time.perf_counter() - tick,
        staging_bytes=sum(p.stat().st_size for c in staged for p in (c.image, c.mask)),
        storage=dict(source_staged=str(target), precomputed=str(work / 'storage/precomputed'),
                     same_storage_parent=str(work / 'storage'), filesystem_device=os.stat(target).st_dev),
        source_metadata_hashes=original_hashes, dataset_provenance={k: v for k, v in exported.items() if k != 'files'},
        identity=identity)
    # Convert NIfTI scalar zoom types into ordinary JSON numbers.
    for record in records:
        record['spacing_mm'] = list(map(float, record['spacing_mm']))
    write_json(work / 'b3a_pilot.json', pilot)
    manifest = export_volume_tiles(staged, config, parts, work / 'storage/precomputed',
        provenance=pilot, expected_hashes=entries)
    validate_export(manifest)
    shutil.copyfile(manifest, work / 'b3a_precompute_manifest.json')
    write_json(work / 'b3a_precompute_manifest_sha.json', dict(sha256=file_hash(manifest)))
    return staged, config, manifest, paths, original_hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New experiment directory')
    parser.add_argument('--synthetic-xml', type=Path, required=True, help='PASS XML from tests/test_b3a_delivery.py')
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--phase', choices=('all', 'prepare', 'correctness', 'benchmark'), default='all',
                        help='Separate phases reuse the validated pilot export, never export it twice')
    args = parser.parse_args(argv)
    if args.repetitions < 3 or args.batch_size < 1 or any(w not in (0, 2) for w in args.workers):
        parser.error('Need >=3 measured runs, positive batch size, workers0/2')
    torch.set_num_threads(1)
    work = args.output_dir.resolve()
    if work.is_relative_to(args.data_root.resolve()):
        parser.error('Output must be outside the source dataset')
    if args.phase in ('all', 'prepare'):
        work.mkdir(parents=True, exist_ok=False)
        write_json(work / 'b3a_environment.json', dict(environment=environment(), source=source_versions()))
    elif not work.is_dir():
        parser.error('Existing prepared experiment required for correctness/benchmark phase')
    try:
        if args.phase in ('all', 'prepare'):
            cases, config, manifest, paths, hashes = prepare(args.data_root.resolve(), work)
        else:
            _, _, config, _, _, paths, _ = load_inputs(args.data_root.resolve(), 'cases.json',
                'splits_covid19_20_5fold.json', 'merit_precomputed_v3.yaml', 'EXPORT_COMPLETE.json')
            pilot = json.loads((work / 'b3a_pilot.json').read_text())
            hashes = pilot['source_metadata_hashes']
            cases = [Case(c['case_id'], Path(c['staged_paths'][0]), Path(c['staged_paths'][1]), c['group_id'])
                     for c in pilot['cases']]
            manifest = work / 'storage/precomputed/b3a_precompute_manifest.json'
            validate_export(manifest)
        if args.phase in ('all', 'correctness'):
            if (work / 'b3a_correctness.json').exists():
                raise FileExistsError('Correctness report already exists; preserve previous evidence')
            correctness_a0003(cases, config, manifest, work / 'b3a_correctness.json', args.synthetic_xml)
        if args.phase in ('all', 'benchmark'):
            if (work / 'b3a_benchmark.json').exists():
                raise FileExistsError('Benchmark report already exists; preserve previous evidence')
            benchmark(cases, config, manifest, work, args.workers, args.batch_size, args.repetitions)
        after = {name: file_hash(path) for name, path in paths.items()}
        if after != hashes:
            raise ValueError('Original V3 metadata changed')
        write_json(work / 'b3a_input_metadata_unchanged.json', dict(status='PASS', before=hashes, after=after))
    except Exception as error:
        write_json(work / 'B3A_ERROR.json', dict(status='FAIL', error=f'{type(error).__name__}: {error}'))
        print(f'STOP B3A: {error}', file=sys.stderr)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
