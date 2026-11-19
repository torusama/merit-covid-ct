"""Short, controlled fast-execution tests of BatchNorm adaptation and case order.

All arms start from the same epoch-0 weights, optimizer moments and RNG. No
production checkpoint is changed. Scores are fixed, label-selected slice probes,
not full-volume validation. BN-only arms never run backward or optimizer.step.
"""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

import torch


def case_pools(dataset, loader_state, per_case):
    """Replay the native loader base seed and first two volume permutations."""
    generator = torch.Generator()
    generator.set_state(loader_state)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    order = torch.randperm(len(dataset.ends)-1, generator=generator).tolist()
    if len(order) < 2:
        raise ValueError('Case-order control requires two training volumes')
    pools = []
    for case_index in order[:2]:
        begin, end = dataset.ends[case_index:case_index+2]
        if end-begin < per_case:
            raise ValueError('Training volume is shorter than the requested control')
        permutation = torch.randperm(end-begin, generator=generator).tolist()
        pools.append([begin+index for index in permutation[:per_case]])
    return pools


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bn_snapshot(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for key in ('running_mean', 'running_var', 'num_batches_tracked'):
                value = getattr(module, key)
                if value is not None:
                    result[name+'.'+key] = value.detach().cpu().clone()
    return result


def bn_drift(before, after):
    assert before.keys() == after.keys()
    changes = []
    for key, value in before.items():
        if not torch.equal(value, after[key]):
            changes.append(key)
    ratios, shifts, counter_changes = [], [], []
    for key, value in before.items():
        if key.endswith('running_var'):
            ratios.append(float((after[key]/value.clamp_min(1e-12)).max()))
            mean_key = key.replace('running_var', 'running_mean')
            shifts.append(float(((after[mean_key]-before[mean_key]).abs()/value.sqrt().clamp_min(1e-6)).max()))
        elif key.endswith('num_batches_tracked'):
            counter_changes.append(int(after[key]-value))
    return {'changed_buffers': len(changes), 'max_variance_ratio': max(ratios, default=1.),
            'max_mean_shift_old_std': max(shifts, default=0.),
            'batch_counter_increments': sorted(set(counter_changes))}


def freeze_bn(model):
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def independent_resume(trainer, checkpoint):
    # Adam's CPU step tensors can otherwise alias the input state even on CUDA.
    # On CPU, moments can alias as well. Every arm needs a pristine optimizer.
    independent = dict(checkpoint)
    independent['optimizer'] = deepcopy(checkpoint['optimizer'])
    independent['scheduler'] = deepcopy(checkpoint['scheduler'])
    trainer.resume(independent)


def execute_arm(trainer, batches, *, update_weights, frozen_bn):
    """Only this helper performs the bounded forward/backward operations."""
    rows = []
    trainer.model.train()
    for index, batch in enumerate(batches):
        if frozen_bn:
            freeze_bn(trainer.model)
        trainer.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(update_weights):
            with torch.autocast(trainer.device.type, dtype=torch.float16, enabled=trainer.amp_enabled):
                result = trainer.model(batch['image_slab'].to(trainer.device), batch['master_index'],
                                       batch['z_offsets_mm'], batch.get('attention_valid'))
                losses = trainer.criterion(result, batch['mask_slab'].to(trainer.device), batch['master_index'],
                    valid_region=batch['valid_region'].to(trainer.device), z_offsets_mm=batch['z_offsets_mm'])
            if not all(bool(torch.isfinite(value).all()) for value in losses.values()):
                raise FloatingPointError('Nonfinite loss in controlled BatchNorm test')
            row = {'batch': index+1, 'case': batch['case_id'][0],
                   'master': int(batch['master_slice_index'][0]),
                   'master_lesion_voxels': int(batch['master_mask'].sum()),
                   'losses': {key:float(value.detach()) for key,value in losses.items()},
                   'optimizer_update': False}
            if update_weights:
                trainer.scaler.scale(losses['total']).backward()
                trainer.scaler.unscale_(trainer.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.settings['gradient_clip'])
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError('Nonfinite gradients in controlled BatchNorm test')
                scale = trainer.scaler.get_scale()
                trainer.scaler.step(trainer.optimizer); trainer.scaler.update()
                if trainer.scaler.get_scale() < scale:
                    raise FloatingPointError('AMP update skipped; matched-step comparison is invalid')
                row.update(optimizer_update=True, gradient_norm=float(norm))
        rows.append(row)
        print('ARM BATCH', json.dumps(row), flush=True)
        del result, losses
    trainer.optimizer.zero_grad(set_to_none=True)
    return rows


def run(checkpoint_path, history_path, expected_hashes, output, *, per_case=10,
        learning_rate=3e-5, probe_case_ids=None, expected_first_batches=None):
    from merit_covid.engine.training import Trainer, split_datasets, validate_experiment
    from merit_covid.data.pipeline.config import PreprocessConfig
    from merit_covid.data.pipeline.dataloader import make_dataloader, collate_slabs
    from scripts.kaggle.resume_checkpoint import verify_dataset_content
    from tools.epoch1_low_lr_pilot import branch_checkpoint, make_probes, evaluate_probes

    if type(per_case) is not int or per_case < 1:
        raise ValueError('per_case must be a positive integer')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/'audit-report.json').exists():
        raise FileExistsError('Choose a fresh output directory')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    history = json.loads(Path(history_path).read_text().strip())
    if history['epoch'] != 0 or history['best_dice'] != checkpoint['best_metric']:
        raise ValueError('Parent checkpoint/history mismatch')
    base = deepcopy(checkpoint['config'])
    base['training'].update(epochs=2, deterministic=False, factorized_mer=True, mer_recomputation=False)
    base['model']['query_chunk_size'] = 4096
    base = validate_experiment(base)
    if base['training']['augmentation_flips']:
        raise ValueError('Control expects the original non-augmented training protocol')
    target = deepcopy(base); target['training']['learning_rate'] = learning_rate
    target = validate_experiment(target)
    datasets, provenance = split_datasets(target, 0)
    provenance['train_order'] = 'seeded case shuffle, then seeded master shuffle within case; every master once'
    verify_dataset_content(sorted([case for ds in datasets.values() for case in ds.cases], key=lambda c:c.case_id), expected_hashes)
    loader = make_dataloader(datasets['train'], batch_size=1, shuffle=True, shuffle_mode='volume',
                             seed=target['training']['seed'], num_workers=0, samples_per_volume=0)
    trainer = Trainer(target, loader, datasets['validation'].cases, provenance, output/'unused-checkpoint-directory')
    migrated = branch_checkpoint(checkpoint, base, trainer.config, trainer.provenance)
    trainer.resume(migrated)
    probe_case_ids = probe_case_ids or sorted(case.case_id for case in datasets['validation'].cases)[:1]
    probes = make_probes(datasets['validation'].cases, PreprocessConfig(**target['preprocess']), probe_case_ids)
    index_pools = case_pools(datasets['train'], checkpoint['loader_generator'], per_case)
    pools = [[collate_slabs([datasets['train'][index]]) for index in indices] for indices in index_pools]
    datasets['train']._cached = None; datasets['train']._cached_index = None
    first_ids = [[batch['case_id'][0], int(batch['master_slice_index'][0])] for batch in pools[0]]
    if expected_first_batches is not None and first_ids != expected_first_batches:
        raise ValueError('Reference batch order differs from the completed low-LR pilot')
    blocked = pools[0]+pools[1]
    interleaved = [batch for pair in zip(*pools) for batch in pair]
    identity = lambda batch: (batch['case_id'][0], int(batch['master_slice_index'][0]))
    assert sorted(map(identity, blocked)) == sorted(map(identity, interleaved))
    trainer.model.train(); trainer.model.set_epoch(1, warmup_epochs=trainer.settings['warmup_epochs'])
    initial_hash = parameter_hash(trainer.model)
    initial_bn = bn_snapshot(trainer.model)
    baseline = evaluate_probes(trainer, probes, 0)
    report = {'scope': __doc__, 'parent_epoch': 0, 'learning_rate': learning_rate,
        'per_case_batches': per_case, 'initial_parameter_sha256': initial_hash,
        'reference_batch_order': first_ids, 'two_case_pool': list(map(identity, blocked)),
        'baseline': baseline, 'arms': [],
        'limits': 'Short slice diagnostics; BN-only order arms use the exact same CT/slab pool in a different order. Not full-volume validation or full-epoch training equivalence.'}
    specifications = [
        ('reference_train_blocked', pools[0], True, False),
        ('bn_only_blocked', pools[0], False, False),
        ('frozen_bn_train_blocked', pools[0], True, True),
        ('bn_only_two_cases_blocked', blocked, False, False),
        ('bn_only_two_cases_interleaved', interleaved, False, False),
    ]
    (output/'audit-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    start = time.monotonic()
    try:
        for name, batches, update_weights, frozen_bn in specifications:
            print('ARM START', name, flush=True)
            trainer.optimizer.zero_grad(set_to_none=True)
            independent_resume(trainer, migrated)
            trainer.model.train(); trainer.model.set_epoch(1, warmup_epochs=trainer.settings['warmup_epochs'])
            assert parameter_hash(trainer.model) == initial_hash
            reset_bn = bn_snapshot(trainer.model)
            assert all(torch.equal(value, reset_bn[key]) for key,value in initial_bn.items())
            before = time.monotonic()
            try:
                rows = execute_arm(trainer, batches, update_weights=update_weights, frozen_bn=frozen_bn)
            except FloatingPointError as error:
                report['arms'].append({'name': name, 'status': 'numerical_failure', 'error': str(error),
                    'samples_requested': len(batches), 'frozen_bn': frozen_bn, 'seconds': time.monotonic()-before})
                (output/'audit-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
                print('ARM FAILED', name, str(error), flush=True)
                continue
            after_hash = parameter_hash(trainer.model)
            unchanged = after_hash == initial_hash
            if not update_weights and not unchanged:
                raise AssertionError('BN-only arm changed model parameters')
            after_bn = bn_snapshot(trainer.model)
            drift = bn_drift(initial_bn, after_bn)
            if frozen_bn and drift['changed_buffers'] != 0:
                raise AssertionError('Frozen-BN arm changed running statistics')
            metrics = evaluate_probes(trainer, probes, len(batches))
            arm = {'name': name, 'status': 'complete', 'samples': len(batches), 'optimizer_steps': sum(row['optimizer_update'] for row in rows),
                'parameters_unchanged': unchanged, 'parameter_sha256': after_hash,
                'frozen_bn': frozen_bn, 'bn_drift': drift, 'probe': metrics, 'batches': rows,
                'seconds': time.monotonic()-before}
            report['arms'].append(arm)
            report['elapsed_seconds'] = time.monotonic()-start
            (output/'audit-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
            print('ARM RESULT', json.dumps({k:v for k,v in arm.items() if k not in ('batches','probe')}), flush=True)
            gc.collect()
        status = {'status': 'bn_volume_audit_complete', 'arms': len(specifications),
            'optimizer_steps_in_completed_arms': sum(arm.get('optimizer_steps', 0) for arm in report['arms']),
            'failed_arms': sum(arm.get('status') != 'complete' for arm in report['arms']),
            'production_checkpoints_changed': False, 'full_validation_performed': False}
    except BaseException as error:
        status = {'status': 'audit_error', 'error': str(error), 'completed_arms': len(report['arms'])}
        (output/'run-status.json').write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
        raise
    (output/'run-status.json').write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
    print('FINAL STATUS', json.dumps(status), flush=True)
    return report
