"""Bounded read-only probes separating attention, PCG and BatchNorm effects.

Probe slices are selected with validation labels for diagnosis, not for reporting
an unbiased validation score. No optimizer is constructed and no weights are saved.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path
import time
import types

import numpy as np
import torch


def gradient_probe(model, checkpoint, cases, preprocess, output, *, require_positive=False):
    """Compare one native-grid AMP backward pass, resetting identical state/RNG."""
    from merit_covid.data.pipeline.dataset import CTSlabDataset
    from merit_covid.data.pipeline.dataloader import make_dataloader
    from merit_covid.engine.activation_checkpointing import checkpointed_mer
    from merit_covid.engine.factorized_attention import factorized_mer
    from merit_covid.models.merit_unetrpp import MERITLoss
    ids = set(checkpoint['provenance']['partitions']['train']['case_ids'])
    dataset = CTSlabDataset([case for case in cases if case.case_id in ids], preprocess)
    dataset.partition = 'train'
    loader = make_dataloader(dataset, batch_size=1, shuffle=True, shuffle_mode='volume', seed=42, num_workers=0)
    loader.generator.set_state(checkpoint['loader_generator'])
    iterator = iter(loader)
    batch = next(iterator)
    if require_positive:
        while not bool(batch['master_mask'].any()):
            batch = next(iterator)
    criterion = MERITLoss(**checkpoint['config']['loss'])
    scale = float(checkpoint['scaler']['scale'])
    reference = None
    rows = []
    for fast in (False, True):
        model.load_state_dict(checkpoint['model'])
        model.train(); model.set_epoch(1, warmup_epochs=30)
        model.zero_grad(set_to_none=True)
        model.mer.forward = types.MethodType(factorized_mer if fast else checkpointed_mer, model.mer)
        model.mer.query_chunk_size = 4096 if fast else 128
        torch.set_rng_state(checkpoint['rng']['torch'])
        torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        start = time.monotonic()
        with torch.autocast('cuda', dtype=torch.float16):
            result = model(batch['image_slab'].cuda(), batch['master_index'], batch['z_offsets_mm'], batch.get('attention_valid'))
            losses = criterion(result, batch['mask_slab'].cuda(), batch['master_index'],
                               valid_region=batch['valid_region'].cuda(), z_offsets_mm=batch['z_offsets_mm'])
        (losses['total'] * scale).backward()
        grads = {name: p.grad.detach().float().cpu() / scale for name, p in model.named_parameters() if p.grad is not None}
        row = {'variant': 'fast' if fast else 'original', 'case': batch['case_id'][0],
               'master': int(batch['master_slice_index'][0]), 'losses': {k: float(v.detach()) for k,v in losses.items()},
               'master_lesion_voxels': int(batch['master_mask'].sum()),
               'mer_gradient_l2': sum(float(v.square().sum()) for k,v in grads.items() if k.startswith('mer.'))**0.5,
               'finite_gradients': all(bool(torch.isfinite(v).all()) for v in grads.values()),
               'seconds': time.monotonic() - start}
        if not fast:
            reference = grads
        else:
            assert grads.keys() == reference.keys()
            square, diff, dot, candidate_square = 0., 0., 0., 0.
            for name, a in reference.items():
                b = grads[name]
                square += float(a.square().sum()); candidate_square += float(b.square().sum())
                diff += float((a-b).square().sum()); dot += float((a*b).sum())
            row['gradient_relative_l2'] = (diff/max(square, 1e-30))**0.5
            row['gradient_cosine'] = dot / max((square*candidate_square)**0.5, 1e-30)
            mer_square = sum(float(v.square().sum()) for k,v in reference.items() if k.startswith('mer.'))
            mer_diff = sum(float((v-grads[k]).square().sum()) for k,v in reference.items() if k.startswith('mer.'))
            row['mer_gradient_relative_l2'] = (mer_diff/max(mer_square, 1e-30))**0.5
        rows.append(row)
        print('FULL MODEL GRADIENT', json.dumps(row), flush=True)
        del result, losses
        model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
    (output/'native-gradient-audit.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    return rows


def run_positive_gradient(checkpoint0, checkpoint1, manifest, output):
    from merit_covid.data.pipeline.cases import load_cases
    from merit_covid.data.pipeline.config import PreprocessConfig
    from merit_covid.engine.activation_checkpointing import configure_model_execution
    from merit_covid.models.merit_unetrpp import MERITUNETRPP
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    cp0 = torch.load(checkpoint0, map_location='cpu', mmap=True, weights_only=False)
    cp1 = torch.load(checkpoint1, map_location='cpu', mmap=True, weights_only=False)
    assert [cp0['epoch'], cp1['epoch']] == [0, 1]
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    cases, _ = load_cases(manifest)
    model = MERITUNETRPP(**cp0['config']['model']).cuda()
    configure_model_execution(model, cp1['config']['training'])
    result = gradient_probe(model, cp0, cases, PreprocessConfig(**cp0['config']['preprocess']), output, require_positive=True)
    print('FINAL STATUS: POSITIVE_GRADIENT_AUDIT_COMPLETE', flush=True)
    return result


def summarize(probability, truth):
    prediction = probability >= 0.5
    tp = int((prediction & truth).sum())
    predicted, actual = int(prediction.sum()), int(truth.sum())
    return {'dice': 2*tp/(predicted+actual) if predicted+actual else 1.0,
        'predicted_voxels': predicted, 'truth_voxels': actual,
        'recall': tp/actual if actual else None, 'probability_mean': float(probability.mean()),
        'probability_max': float(probability.max()),
        'positive_gt_probability_mean': float(probability[truth].mean()) if actual else None}


def run(checkpoint0, checkpoint1, manifest, output):
    from merit_covid.data.pipeline.cases import load_cases
    from merit_covid.data.pipeline.config import PreprocessConfig
    from merit_covid.data.pipeline.dataset import CTSlabDataset
    from merit_covid.data.pipeline.dataloader import collate_slabs
    from merit_covid.engine.activation_checkpointing import configure_model_execution, checkpointed_mer
    from merit_covid.engine.factorized_attention import factorized_mer
    from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss

    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    cp0 = torch.load(checkpoint0, map_location='cpu', mmap=True, weights_only=False)
    cp1 = torch.load(checkpoint1, map_location='cpu', mmap=True, weights_only=False)
    assert [cp0['epoch'], cp1['epoch']] == [0, 1]
    assert not cp0.get('in_progress') and not cp1.get('in_progress')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    cases, _ = load_cases(manifest)
    ids = sorted(cp0['provenance']['partitions']['validation']['case_ids'])[:3]
    assert cp0['provenance']['partitions'] == cp1['provenance']['partitions']
    selected = [case for case in cases if case.case_id in ids]
    preprocess = PreprocessConfig(**cp0['config']['preprocess'])
    probes = []
    for case in selected:
        dataset = CTSlabDataset([case], preprocess)
        _, mask, _ = dataset._load_case(0)
        counts = mask.numpy().sum(axis=(1, 2))
        masters = sorted({int(np.argmax(counts)), 0})
        for master in masters:
            batch = collate_slabs([dataset[master]])
            local = int(batch['master_index'][0])
            probes.append({'case': case.case_id, 'master': master, 'batch': batch,
                           'truth': batch['mask_slab'][0, 0, local].numpy() > 0})
        del dataset, mask
    model = MERITUNETRPP(**cp0['config']['model']).cuda()
    settings = deepcopy(cp1['config']['training'])
    configure_model_execution(model, settings)
    criterion = MERITLoss(**cp0['config']['loss'])
    bn = [(name, module) for name, module in model.named_modules()
          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    report = {'scope': __doc__, 'gpu': torch.cuda.get_device_name(),
              'batchnorm_layers': len(bn), 'probes': [], 'attention_comparisons': []}
    probabilities = {}
    variants = [('epoch0', cp0, 'fast'), ('epoch0', cp0, 'original'),
                ('epoch1', cp1, 'fast'), ('epoch1', cp1, 'original'),
                ('epoch1', cp1, 'batch_stats'), ('epoch1', cp1, 'epoch0_stats'),
                ('epoch1', cp1, 'amp'), ('epoch1', cp1, 'amp_batch_stats')]
    for epoch, checkpoint, variant in variants:
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        model.mer.forward = types.MethodType(checkpointed_mer if variant == 'original' else factorized_mer, model.mer)
        model.mer.query_chunk_size = 128 if variant == 'original' else 4096
        if variant in ('batch_stats', 'amp_batch_stats'):
            for _, module in bn:
                module.train(); module.momentum = 0.0
        elif variant == 'epoch0_stats':
            with torch.no_grad():
                for name, module in bn:
                    module.running_mean.copy_(cp0['model'][name + '.running_mean'])
                    module.running_var.copy_(cp0['model'][name + '.running_var'])
        for i, probe in enumerate(probes):
            batch = probe['batch']; start = time.monotonic()
            print('PROBE START', epoch, variant, probe['case'], probe['master'], flush=True)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=variant.startswith('amp')):
                result = model(batch['image_slab'].cuda(), batch['master_index'], batch['z_offsets_mm'], batch.get('attention_valid'))
                losses = criterion(result, batch['mask_slab'].cuda(), batch['master_index'],
                                   valid_region=batch['valid_region'].cuda(), z_offsets_mm=batch['z_offsets_mm'])
                local = int(batch['master_index'][0])
                corrected = (result.logits[0, 1, local] - result.logits[0, 0, local]).sigmoid().cpu().numpy()
                raw = (result.raw_logits[0, 1, local] - result.raw_logits[0, 0, local]).sigmoid().cpu().numpy()
                presence = float(result.presence[0, local])
            row = {'epoch': epoch, 'variant': variant, 'case': probe['case'], 'master': probe['master'],
                   'pcg_ramp': float(model.pcg_ramp), 'presence': presence,
                   'corrected': summarize(corrected, probe['truth']), 'raw': summarize(raw, probe['truth']),
                   'seconds': time.monotonic() - start, 'losses': {k: float(v) for k,v in losses.items()}}
            report['probes'].append(row)
            print('PROBE RESULT', json.dumps(row), flush=True)
            if variant == 'fast':
                probabilities[(epoch, i)] = corrected
            elif variant == 'original':
                difference = np.abs(corrected - probabilities[(epoch, i)])
                report['attention_comparisons'].append({'epoch': epoch, 'case': probe['case'], 'master': probe['master'],
                    'max_abs_probability_difference': float(difference.max()),
                    'mean_abs_probability_difference': float(difference.mean())})
            (output/'partial.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            del result, corrected, raw, losses
            gc.collect(); torch.cuda.empty_cache()
    report['native_gradient_audit'] = gradient_probe(model, cp0, cases, preprocess, output)
    (output/'inference-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('FINAL STATUS: READ_ONLY_AUDIT_COMPLETE', flush=True)
    return report
