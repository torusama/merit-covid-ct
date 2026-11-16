"""Explicit, audited epoch-boundary execution migration and one-epoch continuation."""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import torch


def migrate_execution(checkpoint, target_config, current_provenance):
    from merit_covid.engine.training import protocol_fingerprint, validate_experiment
    from scripts.kaggle.resume_checkpoint import rebind_verified_timestamps
    if checkpoint.get('schema_version')!=2 or checkpoint.get('epoch')!=0 or checkpoint.get('in_progress'):
        raise ValueError('Migration requires completed schema-2 epoch 0')
    old=validate_experiment(checkpoint['config'])
    expected=deepcopy(old)
    expected['training'].update(epochs=2,deterministic=False,factorized_mer=True,mer_recomputation=False)
    expected['model']['query_chunk_size']=4096
    target=validate_experiment(target_config)
    if target!=expected:
        raise ValueError('Migration contains an unapproved protocol change')
    if not old['training']['activation_checkpointing'] or not old['model']['use_mer']:
        raise ValueError('Expected original checkpointed MER')
    # Verify old metadata and permit ONLY data timestamps to vary across Kaggle VMs.
    rebound=rebind_verified_timestamps(checkpoint,checkpoint['config'],current_provenance,
        protocol_fingerprint(checkpoint['config'],current_provenance))
    migrated=rebound.copy()
    migrated['config']=target
    migrated['protocol_fingerprint']=protocol_fingerprint(target,current_provenance)
    migrated['execution_migration']=dict(from_epoch=0,next_epoch=1,
        old_fingerprint=checkpoint['protocol_fingerprint'],new_fingerprint=migrated['protocol_fingerprint'],
        approved_changes={'epochs':2,'deterministic':False,'factorized_mer':True,
                          'mer_recomputation':False,'query_chunk_size':4096},
        optimizer_reset=False,learning_rate_changed=False)
    for key in ('model','optimizer','scheduler','scaler','rng','loader_generator'):
        if migrated[key] is not checkpoint[key]:raise AssertionError('Training state was replaced: '+key)
    return migrated


def run(checkpoint_path,history_path,best_path,config_path,expected_hashes,output):
    from merit_covid.engine.training import Trainer,load_experiment,split_datasets,validate_experiment
    from merit_covid.data.pipeline.dataloader import make_dataloader
    from scripts.kaggle.resume_checkpoint import verify_dataset_content
    from merit_covid.engine.preflight import _state_hash
    from merit_covid.engine import training
    from merit_covid.models.merit_unetrpp import MERITUNETRPPAdapter

    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cp=torch.load(checkpoint_path,map_location='cpu',mmap=True,weights_only=False)
    history=json.loads(Path(history_path).read_text().strip())
    if history['epoch']!=0 or history['best_dice']!=cp['best_metric']:
        raise ValueError('Checkpoint/history identity mismatch')
    config=load_experiment(config_path)
    # Copy the checkpoint's exact frozen non-execution settings, including paths.
    config=deepcopy(cp['config'])
    config['training'].update(epochs=2,deterministic=False,factorized_mer=True,mer_recomputation=False)
    config['model']['query_chunk_size']=4096
    config=validate_experiment(config)
    datasets,provenance=split_datasets(config,0)
    provenance['train_order']='seeded case shuffle, then seeded master shuffle within case; every master once'
    cases=sorted([case for ds in datasets.values() for case in ds.cases],key=lambda c:c.case_id)
    verify_dataset_content(cases,expected_hashes)
    loader=make_dataloader(datasets['train'],batch_size=1,shuffle=True,shuffle_mode='volume',
                           seed=config['training']['seed'],num_workers=0,samples_per_volume=0)
    trainer=Trainer(config,loader,datasets['validation'].cases,provenance,output/'checkpoints')
    migrated=migrate_execution(cp,trainer.config,trainer.provenance)
    trainer.resume(migrated)
    assert trainer.start_epoch==1 and trainer.best_metric==cp['best_metric']
    original_hash=_state_hash(trainer.model)
    # Check every optimizer moment, scheduler and scaler after real CUDA loading.
    state_ids=cp['optimizer']['param_groups'][0]['params']
    for parameter,state_id in zip(trainer.model.parameters(),state_ids):
        for key in ('exp_avg','exp_avg_sq','step'):
            torch.testing.assert_close(trainer.optimizer.state[parameter][key].cpu(),
                cp['optimizer']['state'][state_id][key],rtol=0,atol=0)
    assert trainer.scheduler.state_dict()==cp['scheduler']
    assert trainer.scaler.state_dict()==cp['scaler']
    print('MIGRATION: exact model/optimizer/scheduler/scaler restored; next epoch=1',flush=True)
    (output/'migration.json').write_text(json.dumps(migrated['execution_migration'],indent=2),encoding='utf-8')
    # Two actual training updates with epoch-1 PCG ramp, then roll all state back.
    trainer.model.train();trainer.model.set_epoch(1,warmup_epochs=trainer.settings['warmup_epochs'])
    smoke=[];iterator=iter(loader)
    for index in range(2):
        batch=next(iterator);trainer.optimizer.zero_grad(set_to_none=True);start=time.monotonic()
        with torch.autocast('cuda',dtype=torch.float16,enabled=trainer.amp_enabled):
            output_model=trainer.model(batch['image_slab'].to(trainer.device),batch['master_index'],
                batch['z_offsets_mm'],batch.get('attention_valid'))
            losses=trainer.criterion(output_model,batch['mask_slab'].to(trainer.device),batch['master_index'],
                valid_region=batch['valid_region'].to(trainer.device),z_offsets_mm=batch['z_offsets_mm'])
        if not all(torch.isfinite(v).all() for v in losses.values()):raise FloatingPointError('Smoke loss')
        trainer.scaler.scale(losses['total']).backward();trainer.scaler.unscale_(trainer.optimizer)
        norm=torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),trainer.settings['gradient_clip'])
        if not torch.isfinite(norm):raise FloatingPointError('Smoke gradients')
        scale=trainer.scaler.get_scale();trainer.scaler.step(trainer.optimizer);trainer.scaler.update()
        if trainer.scaler.get_scale()<scale:raise FloatingPointError('Smoke update skipped')
        row=dict(update=index+1,loss=float(losses['total'].detach()),seconds=time.monotonic()-start,
            pcg_ramp=float(trainer.model.pcg_ramp));smoke.append(row);print('SMOKE',row,flush=True)
        del output_model,losses,batch
    assert _state_hash(trainer.model)!=original_hash
    del iterator
    trainer.resume(migrated)
    trainer.optimizer.zero_grad(set_to_none=True)
    assert _state_hash(trainer.model)==original_hash
    assert torch.equal(loader.generator.get_state(),cp['loader_generator'])
    for parameter,state_id in zip(trainer.model.parameters(),state_ids):
        for key in ('exp_avg','exp_avg_sq','step'):
            torch.testing.assert_close(trainer.optimizer.state[parameter][key].cpu(),cp['optimizer']['state'][state_id][key],rtol=0,atol=0)
    assert trainer.scheduler.state_dict()==cp['scheduler'] and trainer.scaler.state_dict()==cp['scaler']
    (output/'continuation-smoke.json').write_text(json.dumps(dict(status='PASS',updates=smoke,
        state_rolled_back=True,weights_sha256=original_hash,optimizer_scheduler_scaler_restored=True),indent=2),encoding='utf-8')
    print('SMOKE PASS: rolled back both diagnostic updates; starting exact epoch-0 saved state',flush=True)
    gc.collect();torch.cuda.empty_cache()
    # Keep original best and history even if epoch 1 does not improve validation.
    # Trainer.fit creates output_dir too late for these pre-fit copies.
    trainer.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path,trainer.output_dir/'best.pt')
    shutil.copy2(history_path,trainer.output_dir/'history.jsonl')
    original_predict=training.predict_slab_case
    original_logits=MERITUNETRPPAdapter.predict_logits
    counts={}
    def logits_progress(adapter,batch):
        result=original_logits(adapter,batch);case=batch['case_id'][0]
        counts[case]=counts.get(case,0)+len(batch['case_id'])
        if counts[case]==1 or counts[case]%20==0:print('VALIDATION',case,'slices',counts[case],flush=True)
        return result
    def case_progress(model,case,*args,**kwargs):
        print('VALIDATION START',case.case_id,flush=True)
        result=original_predict(model,case,*args,**kwargs)
        print('VALIDATION DONE',case.case_id,flush=True)
        return result
    try:
        with patch.object(training,'predict_slab_case',case_progress),patch.object(MERITUNETRPPAdapter,'predict_logits',logits_progress):
            history_new=trainer.fit(resume=migrated,checkpoint_every=100,max_seconds=None)
        if len(history_new)!=1 or history_new[0]['epoch']!=1:
            raise RuntimeError('Expected exactly one completed epoch: index 1')
        status=dict(status='epoch_complete',epoch=1,checkpoint='checkpoints/last.pt',
            next_epoch=2,training_stopped_after_requested_epoch=True)
    except BaseException as error:
        status=dict(status='training_error',error=str(error),checkpoint='checkpoints/in_progress.pt' if (trainer.output_dir/'in_progress.pt').exists() else None)
        (output/'run-status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        raise
    (output/'run-status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('FINAL STATUS',json.dumps(status),flush=True)
