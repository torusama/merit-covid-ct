"""Validation-only comparison from a trusted completed Kaggle checkpoint.

Run inside the exact frozen source snapshot that produced the checkpoint.
Only the isolated factorized forward overlay is added; no training or migration.
"""
import gc
import hashlib
import json
from pathlib import Path
import time
import types
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch


def compare_reports(candidate, baseline, tolerance=0.001):
    """Reject missing/extra volumes and evaluate per-case drift, not just mean Dice."""
    left={r['case_id']:r for r in baseline['per_case']}
    right={r['case_id']:r for r in candidate['per_case']}
    if len(left)!=len(baseline['per_case']) or len(right)!=len(candidate['per_case']) or set(left)!=set(right):
        raise ValueError('Validation case identities differ or are duplicated')
    changes=[]
    for case_id in sorted(left):
        a,b=left[case_id],right[case_id]
        if a['group_id']!=b['group_id']:raise ValueError('Group identity differs')
        if a['ground_truth_foreground_voxels']!=b['ground_truth_foreground_voxels']:
            raise ValueError('Ground truth differs')
        changes.append(dict(case_id=case_id,**{key:b[key]-a[key] for key in ('dice','iou','precision','recall')}))
    worst=max(abs(r[key]) for r in changes for key in ('dice','iou','precision','recall'))
    return dict(status='PASS' if worst<=tolerance else 'REVIEW_REQUIRED',per_case_tolerance=tolerance,
        max_absolute_metric_difference=worst,per_case_differences=changes,
        macro_differences={key:candidate['macro'][key]['mean']-baseline['macro'][key]['mean']
                           for key in ('dice','iou','precision','recall')})


def run(checkpoint_path, manifest, baseline_path, overlay_source, expected_hashes, output):
    from merit_covid.data.pipeline.cases import load_cases
    from merit_covid.data.pipeline.config import PreprocessConfig
    from merit_covid.data.pipeline.dataset import CTSlabDataset
    from merit_covid.data.pipeline.dataloader import collate_slabs
    from merit_covid.data.pipeline.metrics import aggregate_results
    from merit_covid.models.merit_unetrpp import MERITUNETRPP
    from merit_covid.engine import training
    from merit_covid.engine.preflight import _state_hash
    from scripts.kaggle.resume_checkpoint import verify_dataset_content

    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'validation.json').exists():raise FileExistsError('Existing validation result')
    print('Checking trusted checkpoint metadata and source bytes',flush=True)
    cp=torch.load(checkpoint_path,map_location='cpu',mmap=True,weights_only=False)
    if cp.get('schema_version')!=2 or cp.get('in_progress') or cp['epoch']!=0:
        raise ValueError('Requires the completed schema-2 epoch-0 checkpoint')
    if training.protocol_fingerprint(cp['config'],cp['provenance'])!=cp['protocol_fingerprint']:
        raise ValueError('Checkpoint protocol inconsistent')
    baseline=json.loads(Path(baseline_path).read_text(encoding='utf-8').strip())
    if baseline['epoch']!=cp['epoch'] or baseline['best_dice']!=cp['best_metric']:
        raise ValueError('History and checkpoint differ')
    cases,identity=load_cases(manifest,expected_count=199)
    verify_dataset_content(cases,expected_hashes)
    partitions=cp['provenance']['partitions']
    if partitions['fold']!=0:raise ValueError('Unexpected fold')
    for i,a in enumerate(('train','validation','test')):
        for b in ('train','validation','test')[i+1:]:
            if set(partitions[a]['case_ids'])&set(partitions[b]['case_ids']) or set(partitions[a]['group_ids'])&set(partitions[b]['group_ids']):
                raise ValueError('Partition leakage')
    ids=set(partitions['validation']['case_ids'])
    selected=[c for c in cases if c.case_id in ids]
    if len(selected)!=32 or ids!={r['case_id'] for r in baseline['validation']['per_case']}:
        raise ValueError('Requires exact 32-case validation partition')
    torch.set_num_threads(cp['config']['training']['threads'])
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    model=MERITUNETRPP(**cp['config']['model']).cuda()
    training.configure_model_execution(model,cp['config']['training'])
    model.load_state_dict(cp['model'],strict=True);model.eval()
    initial_hash=_state_hash(model)
    # Preserve checkpoint's PCG buffer. Never call set_epoch or initialize again.
    ramp=float(model.pcg_ramp)
    print('Loaded epoch',cp['epoch'],'PCG ramp',ramp,'state SHA256',initial_hash,flush=True)
    overlay=types.ModuleType('isolated_fast_mer')
    exec(compile(overlay_source,'isolated_fast_mer.py','exec'),overlay.__dict__)
    original_forward=model.mer.forward
    preprocessing=PreprocessConfig(**cp['config']['preprocess'])
    rows=[];probes=[]
    all_gt={c.mask.resolve() for c in cases};original_load=nib.load
    def ct_only_load(filename,*args,**kwargs):
        if Path(filename).resolve() in all_gt:raise AssertionError('Prediction tried to read GT')
        return original_load(filename,*args,**kwargs)
    original_predict=training.predict_slab_case
    def guarded_predict(*args,**kwargs):
        with patch.object(nib,'load',ct_only_load):return original_predict(*args,**kwargs)
    for case in selected[:3]:
        # Fixed middle master, selected before consulting GT or predictions.
        dataset=CTSlabDataset([case],preprocessing,require_mask=False)
        batch=collate_slabs([dataset[len(dataset)//2]])
        image=batch['image_slab'].cuda()
        with patch.object(nib,'load',ct_only_load),torch.inference_mode():
            model.mer.forward=original_forward
            model.mer.query_chunk_size=cp['config']['model']['query_chunk_size']
            torch.use_deterministic_algorithms(True)
            reference=model(image,batch['master_index'],batch['z_offsets_mm'],batch.get('attention_valid')).logits
            reference=(reference[:,1]-reference[:,0]).sigmoid().cpu()
            overlay.enable_factorized_mer(model);model.mer.query_chunk_size=4096
            torch.use_deterministic_algorithms(False)
            prediction=model(image,batch['master_index'],batch['z_offsets_mm'],batch.get('attention_valid')).logits
            prediction=(prediction[:,1]-prediction[:,0]).sigmoid().cpu()
        # Same FP32 policy as original validation; do not mix in AMP changes.
        torch.testing.assert_close(reference,prediction,rtol=1e-4,atol=1e-5)
        probes.append(dict(case_id=case.case_id,max_probability_abs_difference=float((reference-prediction).abs().max())))
        del image,batch,dataset,reference,prediction;gc.collect();torch.cuda.empty_cache()
    (output/'trained-weight-probes.json').write_text(json.dumps(probes,indent=2),encoding='utf-8')
    if _state_hash(model)!=initial_hash:raise AssertionError('Overlay changed weights/buffers')
    overlay.enable_factorized_mer(model);model.mer.query_chunk_size=4096
    torch.use_deterministic_algorithms(False)
    for index,case in enumerate(selected):
        print('VALIDATION',index+1,'/32',case.case_id,flush=True);start=time.monotonic()
        with patch.object(training,'predict_slab_case',guarded_predict):
            result=training.evaluate_model(model,[case],preprocessing,
                threshold=cp['config']['training']['threshold'],include_hd95=cp['config']['training']['include_hd95'],
                batch_size=1,num_workers=0,identity=identity)
        rows.extend(result['per_case'])
        (output/'partial-validation.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
        print('DONE',case.case_id,'seconds',round(time.monotonic()-start,2),'dice',rows[-1]['dice'],'recall',rows[-1]['recall'],flush=True)
    report=aggregate_results(rows,cp['provenance']['identity'])
    comparison=compare_reports(report,baseline['validation'])
    unchanged=_state_hash(model)==initial_hash
    if not unchanged:raise AssertionError('Validation changed model state')
    comparison.update(epoch=cp['epoch'],pcg_ramp=ramp,weights_unchanged=unchanged,
        initial_state_sha256=initial_hash,precision='FP32, matching original validation',
        trained_weights_probes=probes,validation=report,
        limitation='Inference validation only. Does not establish equivalent future optimization trajectories or authorize direct checkpoint resume.')
    (output/'validation.json').write_text(json.dumps(comparison,indent=2,allow_nan=False),encoding='utf-8')
    print('TRAINED CHECKPOINT VALIDATION SUMMARY',json.dumps({k:v for k,v in comparison.items() if k not in ('validation','per_case_differences')},indent=2),flush=True)
    return comparison
