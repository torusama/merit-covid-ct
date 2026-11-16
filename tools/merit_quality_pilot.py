"""Paired learning diagnostic on real TRAIN cases and fixed held-out volumes.

Starts from random initialization. This is NOT an evaluation of the user's
Kaggle checkpoint or evidence of convergence after complete training.
"""
import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.data.pipeline.dataloader import collate_slabs
from merit_covid.data.pipeline.metrics import volume_metrics, aggregate_results
from merit_covid.engine.preflight import _setup, _state_hash, _sync, _memory
from merit_covid.engine.slab_inference import predict_slab_case
from merit_covid.engine.training import load_experiment, validate_experiment
from merit_covid.engine import activation_checkpointing as execution
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss
from tools.merit_execution_tuning import recomputation
from tools.merit_factorized_attention import enable_factorized_mer
from tools.merit_real_case_trial import file_hash


def verify_cases(specs, data, split, partition):
    from merit_covid.data.pipeline.cases import Case
    allowed = split['folds'][0][partition]
    result = []
    for spec in specs:
        if spec['case_id'] not in allowed['case_ids'] or spec['group_id'] not in allowed['group_ids']:
            raise ValueError('Case/group does not belong to '+partition)
        for other in ('train','validation','test'):
            if other != partition and (spec['case_id'] in split['folds'][0][other]['case_ids'] or
                                     spec['group_id'] in split['folds'][0][other]['group_ids']):
                raise ValueError('Partition overlap')
        ct, gt = data/spec['ct_name'], data/spec['gt_name']
        if {'ct':file_hash(ct),'gt':file_hash(gt)} != spec['sha256']:
            raise ValueError('Source bytes changed: '+spec['case_id'])
        result.append(Case(spec['case_id'],ct,gt,spec['group_id']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','data-dir','specs','output','shared'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--variant',choices=('reference','fast'),required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.shared.mkdir(parents=True,exist_ok=True)
    report=dict(status='RUNNING',variant=args.variant,
        scope='24 diagnostic updates from random initialization on 3 TRAIN cases; 2 complete held-out validation volumes',
        kaggle_checkpoint_evaluated=False,full_training=False,steps=[])
    phase_file=args.output.with_suffix('.progress.json')
    def phase(name, **details):
        entry=dict(phase=name,time=time.time(),variant=args.variant,**details)
        temp=phase_file.with_suffix('.tmp')
        temp.write_text(json.dumps(entry),encoding='utf-8');temp.replace(phase_file)
        print('PHASE',args.variant,name,json.dumps(details),flush=True)
    phase('setup')
    config=load_experiment(args.config)
    fast=args.variant=='fast'
    config['training'].update(device='cuda',deterministic=not fast)
    config=validate_experiment(config)
    specs=json.loads(args.specs.read_text(encoding='utf-8'))
    split_path=Path(config['data']['split_path'])
    import hashlib
    split_raw=split_path.read_bytes().replace(b'\r\n',b'\n')
    if hashlib.sha256(split_raw).hexdigest()!=specs['split_sha256_lf']:
        raise ValueError('Frozen split changed')
    split=json.loads(split_raw)
    train=verify_cases(specs['train'],args.data_dir,split,'train')
    validation=verify_cases(specs['validation'],args.data_dir,split,'validation')
    preprocess=PreprocessConfig(**config['preprocess'])
    if preprocess.size_hw is not None or config['model']['img_size'] != [11,512,512]:
        raise ValueError('Requires native 512 input')
    device=_setup(config)
    model=execution.configure_model_execution(MERITUNETRPP(**config['model']).to(device),config['training'])
    if fast:
        enable_factorized_mer(model);model.mer.query_chunk_size=4096
    else:
        model.mer.query_chunk_size=128
    report['initial_state_sha256']=_state_hash(model)
    report['effective_query_chunk_size']=model.mer.query_chunk_size
    report['mer_recomputation']=not fast
    report['threshold']=0.5
    model.train()
    # Explicit stress diagnostic with active PCG, not pretending to be epoch 0.
    model.set_epoch(config['training']['warmup_epochs'],warmup_epochs=config['training']['warmup_epochs'])
    report['pcg_ramp']=float(model.pcg_ramp)
    report['config']=config
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=config['training']['amp_init_scale'])
    criterion=MERITLoss(**config['loss'])
    datasets=[CTSlabDataset([case],preprocess) for case in train]
    schedule=[(i,m) for i,spec in enumerate(specs['train']) for m in spec['masters']]*4
    torch.cuda.reset_peak_memory_stats()
    with recomputation(not fast):
        for index,(i,master) in enumerate(schedule):
            phase('load_batch',step=index+1,case_id=train[i].case_id,master=master)
            load_start=time.perf_counter()
            batch=collate_slabs([datasets[i][master]])
            load_seconds=time.perf_counter()-load_start
            optimizer.zero_grad(set_to_none=True)
            _sync(device);start=time.perf_counter()
            phase('forward_loss',step=index+1,load_seconds=load_seconds)
            with torch.autocast('cuda',dtype=torch.float16):
                output=model(batch['image_slab'].to(device),batch['master_index'],batch['z_offsets_mm'],batch.get('attention_valid'))
                losses=criterion(output,batch['mask_slab'].to(device),batch['master_index'],
                    valid_region=batch['valid_region'].to(device),z_offsets_mm=batch['z_offsets_mm'])
            if not all(torch.isfinite(v).all() for v in losses.values()):raise FloatingPointError('Nonfinite loss')
            phase('backward',step=index+1)
            scaler.scale(losses['total']).backward();scaler.unscale_(optimizer)
            phase('optimizer',step=index+1)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['training']['gradient_clip'])
            scale=scaler.get_scale();scaler.step(optimizer);scaler.update();_sync(device)
            skipped=scaler.get_scale()<scale
            if skipped or not torch.isfinite(norm):raise FloatingPointError('Skipped/nonfinite update')
            row=dict(step=index+1,case_id=train[i].case_id,master=master,
                losses={k:float(v.detach()) for k,v in losses.items()},seconds=time.perf_counter()-start)
            report['steps'].append(row)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            print(args.variant,'TRAIN',index+1,'/24 loss',row['losses']['total'],'compute_seconds',round(row['seconds'],3),'load_seconds',round(load_seconds,3),flush=True)
            del output,losses,batch
    report.update(_memory(device))
    del optimizer,scaler,datasets;gc.collect();torch.cuda.empty_cache()
    model.eval();rows=[];differences=[]
    gt_paths={case.mask.resolve() for case in train+validation}
    original_nib_load=nib.load
    def ct_only_load(filename,*positional,**kwargs):
        if Path(filename).resolve() in gt_paths:
            raise AssertionError('Prediction attempted to read a ground-truth mask')
        return original_nib_load(filename,*positional,**kwargs)
    for case in validation:
        phase('validation',case_id=case.case_id)
        print(args.variant,'VALIDATION full volume',case.case_id,flush=True)
        from merit_covid.models.merit_unetrpp import MERITUNETRPPAdapter
        original_predict=MERITUNETRPPAdapter.predict_logits
        slices_done=[0]
        def progress_predict(adapter,batch):
            result=original_predict(adapter,batch)
            slices_done[0]+=len(batch['image_slab'])
            if slices_done[0]==1 or slices_done[0]%10==0:
                phase('validation',case_id=case.case_id,slices_done=slices_done[0])
            return result
        with patch.object(nib,'load',ct_only_load),patch.object(MERITUNETRPPAdapter,'predict_logits',progress_predict),torch.autocast('cuda',dtype=torch.float16):
            prediction=predict_slab_case(model,case,preprocess)
        native=prediction.to_original_xyz()
        truth=nib.load(case.mask).get_fdata(dtype=np.float32)
        row={'case_id':case.case_id,'group_id':case.group_id,
             **volume_metrics(native,truth,0.5,prediction.metadata['original_affine_mm'],False),
             'gt_voxels':int((truth>0).sum()),'predicted_voxels':int((native>=0.5).sum())}
        rows.append(row)
        baseline=args.shared/(case.case_id+'.npy')
        if not fast:
            np.save(baseline,native)
        else:
            ref=np.load(baseline,mmap_mode='r')
            if ref.shape!=native.shape:raise ValueError('Reference prediction grid differs')
            delta_sum=0.;delta_max=0.;disagreements=0
            for z in range(native.shape[2]):
                diff=np.abs(ref[:,:,z]-native[:,:,z]);delta_sum+=float(diff.sum(dtype=np.float64));delta_max=max(delta_max,float(diff.max()))
                disagreements+=int(((ref[:,:,z]>=0.5)!=(native[:,:,z]>=0.5)).sum())
            differences.append(dict(case_id=case.case_id,probability_mean_abs_diff=delta_sum/native.size,
                probability_max_abs_diff=delta_max,mask_disagreement_fraction=disagreements/native.size))
            del ref
        # Visualization may use labels after complete CT-only prediction.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ct=nib.load(case.image).get_fdata(dtype=np.float32)
        z=int((truth>0).sum(axis=(0,1)).argmax())
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax in axes:ax.imshow(ct[:,:,z].T,cmap='gray',origin='lower');ax.axis('off')
        axes[0].set_title('CT');axes[1].imshow(np.ma.masked_where(truth[:,:,z].T==0,truth[:,:,z].T),cmap='Reds',alpha=.5,origin='lower');axes[1].set_title('Ground truth')
        axes[2].imshow(np.ma.masked_where(native[:,:,z].T<.5,native[:,:,z].T),cmap='Blues',alpha=.5,origin='lower');axes[2].set_title('Prediction threshold 0.5')
        fig.suptitle(args.variant+' '+case.case_id+' | Dice '+str(round(row['dice'],4)))
        fig.savefig(args.output.parent/(args.variant+'_'+case.case_id+'.png'),dpi=120);plt.close(fig)
        print(args.variant,'METRICS',json.dumps(row),flush=True)
        del prediction,native,truth,ct
    report['validation']=aggregate_results(rows,split['identity'])
    report['prediction_differences']=differences
    report['final_state_sha256']=_state_hash(model)
    report['source_files_unchanged']=all(file_hash(args.data_dir/s[k])==s['sha256'][kind]
        for s in specs['train']+specs['validation'] for kind,k in [('ct','ct_name'),('gt','gt_name')])
    if not report['source_files_unchanged']:raise ValueError('Source file changed')
    report['status']='PASS'
    phase('complete')
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('QUALITY WORKER PASS',args.variant,flush=True)


if __name__=='__main__':main()
