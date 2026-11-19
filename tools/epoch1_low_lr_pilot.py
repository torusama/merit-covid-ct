"""Bounded low-LR continuation from epoch 0, with fixed diagnostic probes.

This is a separate experiment. Probe scores are label-selected slice diagnostics,
not full-volume validation scores. Production checkpoints are never overwritten.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch


def branch_checkpoint(checkpoint, base_config, target_config, provenance):
    from merit_covid.engine.training import protocol_fingerprint, validate_experiment
    from scripts.kaggle.fast_epoch1 import migrate_execution
    base = validate_experiment(base_config)
    target = validate_experiment(target_config)
    expected = deepcopy(base)
    expected['training']['learning_rate'] = target['training']['learning_rate']
    if target != expected or not 0 < target['training']['learning_rate'] < base['training']['learning_rate']:
        raise ValueError('Pilot may only lower learning rate beyond the audited execution migration')
    result = migrate_execution(checkpoint, base, provenance).copy()
    rate = target['training']['learning_rate']
    result['config'] = target
    result['protocol_fingerprint'] = protocol_fingerprint(target, provenance)
    result['optimizer'] = dict(checkpoint['optimizer'])
    result['optimizer']['param_groups'] = [dict(group, lr=rate) for group in checkpoint['optimizer']['param_groups']]
    for group in result['optimizer']['param_groups']:
        if 'initial_lr' in group:
            group['initial_lr'] = rate
    result['scheduler'] = deepcopy(checkpoint['scheduler'])
    result['scheduler']['_last_lr'] = [rate for _ in result['optimizer']['param_groups']]
    result['execution_migration'] = deepcopy(result['execution_migration'])
    result['execution_migration'].update(new_fingerprint=result['protocol_fingerprint'],
        learning_rate_changed=True, experiment='bounded_low_lr_pilot',
        learning_rate_change={'from': base['training']['learning_rate'], 'to': rate},
        optimizer_moments_preserved=True, full_epoch_complete=False)
    assert result['optimizer']['state'] is checkpoint['optimizer']['state']
    for key in ('model', 'scaler', 'rng', 'loader_generator'):
        assert result[key] is checkpoint[key]
    return result


def make_probes(cases, preprocess, case_ids):
    from merit_covid.data.pipeline.dataset import CTSlabDataset
    from merit_covid.data.pipeline.dataloader import collate_slabs
    selected = {case.case_id: case for case in cases}
    if not case_ids or any(case_id not in selected for case_id in case_ids):
        raise ValueError('Every probe must belong to the validation partition')
    probes = []
    for case_id in case_ids:
        dataset = CTSlabDataset([selected[case_id]], preprocess)
        _, mask, _ = dataset._load_case(0)
        counts = mask.numpy().sum(axis=(1, 2))
        masters = [('positive', int(np.argmax(counts)))] if counts.max() > 0 else []
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            masters.append(('empty', int(empty[0])))
        for kind, master in masters:
            batch = collate_slabs([dataset[master]])
            local = int(batch['master_index'][0])
            probes.append({'case': case_id, 'master': master, 'kind': kind, 'batch': batch,
                           'truth': batch['mask_slab'][0, 0, local].numpy() > 0})
        del dataset, mask
    if not any(probe['kind'] == 'positive' for probe in probes):
        raise ValueError('Pilot needs a positive diagnostic control')
    return probes


def evaluate_probes(trainer, probes, step):
    from tools.audit_epoch1_inference import summarize
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if trainer.device.type == 'cuda' else None)
    was_training = trainer.model.training
    rows = []
    trainer.model.eval()
    try:
        with torch.inference_mode():
            for probe in probes:
                batch = probe['batch']
                result = trainer.model(batch['image_slab'].to(trainer.device), batch['master_index'],
                                       batch['z_offsets_mm'], batch.get('attention_valid'))
                local = int(batch['master_index'][0])
                probability = (result.logits[0, 1, local]-result.logits[0, 0, local]).sigmoid().cpu().numpy()
                raw = (result.raw_logits[0, 1, local]-result.raw_logits[0, 0, local]).sigmoid().cpu().numpy()
                rows.append({'case': probe['case'], 'master': probe['master'], 'kind': probe['kind'],
                    **summarize(probability, probe['truth']), 'raw': summarize(raw, probe['truth'])})
                del result, probability, raw
    finally:
        trainer.model.train(was_training)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
    positive = [row for row in rows if row['kind'] == 'positive']
    empty = [row for row in rows if row['kind'] == 'empty']
    report = {'batch': step, 'positive_slice_mean_dice': float(np.mean([r['dice'] for r in positive])),
        'positive_slice_mean_recall': float(np.mean([r['recall'] for r in positive])),
        'positive_slices_with_empty_prediction': sum(r['predicted_voxels'] == 0 for r in positive),
        'empty_slice_false_positive_voxels': sum(r['predicted_voxels'] for r in empty), 'per_slice': rows}
    print('PROBE', json.dumps({k:v for k,v in report.items() if k != 'per_slice'}), flush=True)
    return report


def run(checkpoint_path, history_path, expected_hashes, output, *, total_batches=50,
        evaluate_every=10, learning_rate=3e-5, probe_case_ids=None):
    from merit_covid.engine.training import Trainer, split_datasets, validate_experiment, augment_slab_flips
    from merit_covid.data.pipeline.dataloader import make_dataloader
    from merit_covid.data.pipeline.config import PreprocessConfig
    from scripts.kaggle.resume_checkpoint import verify_dataset_content
    if type(total_batches) is not int or total_batches < 1 or type(evaluate_every) is not int or evaluate_every < 1:
        raise ValueError('Batch limits must be positive integers')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    checkpoints = output/'checkpoints'; checkpoints.mkdir(parents=True, exist_ok=True)
    if (checkpoints/'in_progress.pt').exists():
        raise FileExistsError('Choose a fresh pilot output directory')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    history = json.loads(Path(history_path).read_text().strip())
    if history['epoch'] != 0 or history['best_dice'] != checkpoint['best_metric']:
        raise ValueError('Epoch-0 checkpoint/history identity mismatch')
    base = deepcopy(checkpoint['config'])
    base['training'].update(epochs=2, deterministic=False, factorized_mer=True, mer_recomputation=False)
    base['model']['query_chunk_size'] = 4096
    base = validate_experiment(base)
    target = deepcopy(base); target['training']['learning_rate'] = learning_rate
    target = validate_experiment(target)
    datasets, provenance = split_datasets(target, 0)
    provenance['train_order'] = 'seeded case shuffle, then seeded master shuffle within case; every master once'
    verify_dataset_content(sorted([c for ds in datasets.values() for c in ds.cases], key=lambda c:c.case_id), expected_hashes)
    loader = make_dataloader(datasets['train'], batch_size=1, shuffle=True, shuffle_mode='volume',
                             seed=target['training']['seed'], num_workers=0, samples_per_volume=0)
    if total_batches >= len(loader):
        raise ValueError('This pilot must stop before completing the epoch')
    trainer = Trainer(target, loader, datasets['validation'].cases, provenance, checkpoints)
    migrated = branch_checkpoint(checkpoint, base, trainer.config, trainer.provenance)
    trainer.resume(migrated)
    assert trainer.start_epoch == 1 and trainer.optimizer.param_groups[0]['lr'] == learning_rate
    probe_case_ids = probe_case_ids or sorted(c.case_id for c in datasets['validation'].cases)[:1]
    probes = make_probes(datasets['validation'].cases, PreprocessConfig(**target['preprocess']), probe_case_ids)
    trainer.model.train(); trainer.model.set_epoch(1, warmup_epochs=trainer.settings['warmup_epochs'])
    report = {'scope': __doc__, 'parent_epoch': 0, 'epoch': 1, 'requested_batches': total_batches,
        'learning_rate': learning_rate, 'execution_migration': migrated['execution_migration'],
        'probe_selection': 'Predeclared validation cases; max-GT-area positive slice and first empty slice. Diagnostic only.',
        'train': [], 'probes': [evaluate_probes(trainer, probes, 0)]}
    (output/'pilot-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    epoch_generator = loader.generator.get_state().clone()
    iterator = iter(loader)
    totals, examples, steps, skipped = {}, 0, 0, 0
    start = time.monotonic()
    try:
        for index in range(total_batches):
            batch = next(iterator)
            if trainer.settings['augmentation_flips']:
                batch = augment_slab_flips(batch)
            trainer.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(trainer.device.type, dtype=torch.float16, enabled=trainer.amp_enabled):
                result = trainer.model(batch['image_slab'].to(trainer.device), batch['master_index'],
                                       batch['z_offsets_mm'], batch.get('attention_valid'))
                losses = trainer.criterion(result, batch['mask_slab'].to(trainer.device), batch['master_index'],
                    valid_region=batch['valid_region'].to(trainer.device), z_offsets_mm=batch['z_offsets_mm'])
            if not all(bool(torch.isfinite(v).all()) for v in losses.values()):
                raise FloatingPointError('Nonfinite pilot loss')
            trainer.scaler.scale(losses['total']).backward()
            trainer.scaler.unscale_(trainer.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.settings['gradient_clip'])
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError('Nonfinite pilot gradients')
            scale = trainer.scaler.get_scale()
            trainer.scaler.step(trainer.optimizer); trainer.scaler.update()
            overflow = trainer.scaler.get_scale() < scale
            skipped += int(overflow); steps += int(not overflow); examples += 1
            values = {k:float(v.detach()) for k,v in losses.items()}
            for key, value in values.items():
                totals[key] = totals.get(key, 0.) + value
            row = {'batch': index+1, 'case': batch['case_id'][0], 'master': int(batch['master_slice_index'][0]),
                   'master_lesion_voxels': int(batch['master_mask'].sum()), 'losses': values,
                   'gradient_norm': float(norm), 'amp_skipped': overflow}
            report['train'].append(row)
            print('PILOT TRAIN', json.dumps(row), flush=True)
            del result, losses, batch
            if (index+1) % evaluate_every == 0 or index+1 == total_batches:
                progress = {'epoch': 1, 'batches': index+1, 'epoch_generator': epoch_generator,
                    'totals': totals.copy(), 'examples': examples, 'steps': steps, 'skipped': skipped}
                trainer.save_checkpoint(checkpoints/'in_progress.pt', 1, progress=progress)
                report['probes'].append(evaluate_probes(trainer, probes, index+1))
                report.update(processed_batches=index+1, optimizer_steps=steps, amp_skipped_steps=skipped,
                              elapsed_seconds=time.monotonic()-start)
                (output/'pilot-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
                gc.collect()
        status = {'status': 'pilot_complete', 'processed_batches': total_batches, 'optimizer_steps': steps,
            'epoch': 1, 'full_epoch_complete': False, 'checkpoint': 'checkpoints/in_progress.pt',
            'stopped_after_requested_batches': True, 'full_validation_performed': False}
    except BaseException as error:
        status = {'status': 'pilot_error', 'error': str(error), 'processed_batches': len(report['train'])}
        (output/'run-status.json').write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
        raise
    (output/'run-status.json').write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
    print('FINAL STATUS', json.dumps(status), flush=True)
    return report
