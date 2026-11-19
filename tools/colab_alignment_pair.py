"""Portable, resumable two-arm alignment ablation for the frozen Colab bundle.

Only paths/timestamps/container sizes and unused CUDA-device RNG entries may be
rebound after full decompressed content verification. No input scaling changes.
Snapshots are explicitly partial-epoch experiment artifacts, never native resume.
"""
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import zipfile

import numpy as np
import torch

ARMS=('reference_alignment','alignment_off')
SCHEMA='merit_colab_alignment_pair_v1'


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def stage_dataset(source,destination):
    """Accept a ZIP or directory; stage only under the declared scratch folder."""
    source=Path(source).expanduser().resolve(strict=True)
    destination=Path(destination).resolve()
    if source==destination or destination.is_relative_to(source):
        raise ValueError('Dataset input and scratch destination must be separate')
    destination.mkdir(parents=True,exist_ok=True)
    marker=destination/'staging.json'
    stamp=dict(source=str(source),size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns,
               directory_policy='nifti-only-v2')
    if source.is_dir():
        # Drive directory mtime need not change when files arrive in a nested
        # upload. Do not reuse an incomplete cache just because its root matches.
        inventory=[]
        for path in sorted(source.rglob('*')):
            if path.is_file() and (path.name.endswith('.nii') or path.name.endswith('.nii.gz')):
                stat=path.stat()
                inventory.append((path.relative_to(source).as_posix(),stat.st_size,stat.st_mtime_ns))
        stamp['nifti_files']=len(inventory)
        stamp['inventory_sha256']=hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
        if not inventory:
            examples=[p.name for p in list(source.iterdir())[:8]]
            raise FileNotFoundError(f'Không có file .nii/.nii.gz dưới {source}. '
                f'Các mục đang có: {examples}. Kiểm tra đường dẫn và đợi Drive upload xong.')
    if marker.exists() and json.loads(marker.read_text())==stamp:
        return destination
    if source.is_file():
        if not zipfile.is_zipfile(source):raise ValueError('Dataset archive must be ZIP, or provide a directory')
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                target=(destination/entry.filename).resolve()
                if not target.is_relative_to(destination) or (entry.external_attr>>16)&0o170000==0o120000:
                    raise ValueError('Archive contains an escaping path or symlink')
            archive.extractall(destination)
    else:
        def ignore_non_nifti(directory,names):
            # inputs may also contain the multi-GB checkpoint/history; they are
            # staged separately and must not be copied as part of the dataset.
            return [name for name in names if not (Path(directory)/name).is_dir()
                    and not (name.endswith('.nii') or name.endswith('.nii.gz'))]
        shutil.copytree(source,destination,dirs_exist_ok=True,ignore=ignore_non_nifti)
    write_json(marker,stamp)
    return destination


def find_dataset_root(source,case_ids):
    """Resolve actual CT/seg filenames recursively, independent of folder name."""
    source=Path(source)
    direct={}
    nested={}
    for case_id in case_ids:
        for key,kind in (('image','ct'),('mask','seg')):
            stem=f'{case_id}_{kind}'
            for suffix in ('.nii','.nii.gz'):
                direct[stem+suffix]=(case_id,key)
            nested[(stem+'.nii',case_id+'.nii')]=(case_id,key)
            nested[(stem+'.nii',stem+'.nii')]=(case_id,key)
    matches={(case_id,key):[] for case_id in case_ids for key in ('image','mask')}
    nifti_count=0
    for path in source.rglob('*'):
        if not path.is_file() or not (path.name.endswith('.nii') or path.name.endswith('.nii.gz')):
            continue
        nifti_count+=1
        pair=direct.get(path.name) or nested.get((path.parent.name,path.name))
        if pair is not None:
            matches[pair].append(path)
    missing=[f'{case_id}_{("ct" if key=="image" else "seg")}.nii[.gz]'
             for (case_id,key),paths in matches.items() if not paths]
    duplicates=[(case_id,key,[str(p) for p in paths])
                for (case_id,key),paths in matches.items() if len(paths)>1]
    counts={key:sum(bool(matches[(case_id,key)]) for case_id in case_ids) for key in ('image','mask')}
    print(f'DATASET SCAN: {source}; NIfTI={nifti_count}; '
          f'CT={counts["image"]}/{len(case_ids)}; GT={counts["mask"]}/{len(case_ids)}',flush=True)
    if duplicates:
        raise ValueError('Có nhiều bản CT/seg cho cùng case; đặt DATASET_SOURCE vào đúng một bộ dữ liệu. '
                         +repr(duplicates[:5]))
    if missing:
        raise FileNotFoundError(f'Data chưa đủ hoặc chọn sai đường dẫn {source}: '
            f'CT {counts["image"]}/{len(case_ids)}, GT {counts["mask"]}/{len(case_ids)}. '
            f'Còn thiếu {len(missing)} file, ví dụ: {missing[:8]}. '
            'Không cần thư mục tên Train; hãy đợi upload xong toàn bộ CT/seg.')
    return source,{case_id:{key:matches[(case_id,key)][0] for key in ('image','mask')} for case_id in case_ids}


def bind_portable_checkpoint(checkpoint,config,provenance,*,cuda_devices=None):
    from merit_covid.engine.training import protocol_fingerprint
    if checkpoint.get('schema_version')!=2 or checkpoint.get('epoch')!=0 or checkpoint.get('in_progress'):
        raise ValueError('Expected completed schema-2 epoch 0')
    if checkpoint['protocol_fingerprint']!=protocol_fingerprint(checkpoint['config'],checkpoint['provenance']):
        raise ValueError('Original checkpoint protocol fingerprint is inconsistent')
    old=deepcopy(checkpoint['config']);new=deepcopy(config)
    if set(old['data'])!=set(new['data']):raise ValueError('Data configuration keys changed')
    for key in ('manifest','split_path','patient_map'):
        if key in old['data']:
            old['data'][key]=new['data'][key]
    if old!=new:raise ValueError('Portable migration may change data paths only')
    previous=checkpoint['provenance']
    for key in ('partitions','identity','all_case_ids','all_group_ids','train_order'):
        if previous.get(key)!=provenance.get(key):raise ValueError('Changed split/group identity: '+key)
    identities=lambda p:[(r['case_id'],r['group_id']) for r in p['source_identity']['cases']]
    if identities(previous)!=identities(provenance):raise ValueError('Source case/group mapping changed')
    result=dict(checkpoint,config=deepcopy(config),provenance=deepcopy(provenance),
        protocol_fingerprint=protocol_fingerprint(config,provenance))
    result['portability_migration']=dict(original_protocol_fingerprint=checkpoint['protocol_fingerprint'],
        content_verified='all CT/mask decompressed SHA256 values match the frozen manifest',
        changed_metadata='data paths, source timestamps/container sizes, manifest/split filename bindings',
        cross_runtime_exact_historical_replay_claimed=False)
    if cuda_devices is not None and checkpoint['rng'].get('cuda') is not None:
        states=checkpoint['rng']['cuda']
        if not 1<=cuda_devices<=len(states):raise ValueError('Cannot invent missing CUDA RNG streams')
        result['rng']=dict(checkpoint['rng'],cuda=states[:cuda_devices])
        result['portability_migration']['cuda_rng_streams']=dict(from_count=len(states),to_count=cuda_devices)
    return result


def prepare(project,dataset,checkpoint_path,history_path,expected_hashes,work,*,parent_sha256,source_sha256,
            expected_cases=199,expected_validation_cases=32):
    from merit_covid.data.pipeline.cases import load_cases,manifest_fingerprint
    from merit_covid.data.pipeline.splits import validate_splits
    from merit_covid.engine.training import split_datasets
    from scripts.kaggle.resume_checkpoint import verify_dataset_content
    project=Path(project).resolve();work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    if file_hash(checkpoint_path)!=parent_sha256:raise ValueError('Original epoch-0 checkpoint SHA256 mismatch')
    local=work/'original-epoch0.pt'
    if not local.exists() or file_hash(local)!=parent_sha256:shutil.copy2(checkpoint_path,local)
    checkpoint=torch.load(local,map_location='cpu',mmap=True,weights_only=False)
    history=json.loads(Path(history_path).read_text().strip())
    if history['epoch']!=0 or history['best_dice']!=checkpoint['best_metric']:
        raise ValueError('Checkpoint and original epoch-0 history do not match')
    if len(expected_hashes)!=expected_cases:raise ValueError('Wrong frozen content-hash manifest')
    _,paths=find_dataset_root(dataset,expected_hashes)
    manifest=project/'data/covid19_20/cases.json'
    records=json.loads(manifest.read_text(encoding='utf-8-sig'))
    if {r['id'] for r in records}!=set(expected_hashes):raise ValueError('Source manifest case IDs changed')
    for record in records:
        for key in ('image','mask'):record[key]=str(paths[record['id']][key].resolve())
    write_json(manifest,records)
    cases,_=load_cases(manifest,expected_count=expected_cases)
    split_path=project/'splits/covid19_20_5fold.json'
    split=json.loads(split_path.read_text(encoding='utf-8-sig'))
    split['manifest_fingerprint']=manifest_fingerprint(cases);validate_splits(split,cases);write_json(split_path,split)
    config=deepcopy(checkpoint['config'])
    config['data'].update(manifest=str(manifest),split_path=str(split_path))
    datasets,provenance=split_datasets(config,0)
    provenance['train_order']=checkpoint['provenance']['train_order']
    if len(datasets['validation'].cases)!=expected_validation_cases:raise ValueError('Validation count changed')
    verify_dataset_content(cases,expected_hashes)
    cuda_devices=torch.cuda.device_count() if config['training']['device']!='cpu' else None
    portable=bind_portable_checkpoint(checkpoint,config,provenance,cuda_devices=cuda_devices)
    destination=work/'portable-epoch0.pt';torch.save(portable,destination)
    local_history=work/'epoch0-history.jsonl';local_history.write_text(json.dumps(history)+'\n',encoding='utf-8')
    context=dict(parent=str(destination),history=str(local_history),hashes=expected_hashes,
        work=str(work),parent_sha256=parent_sha256,source_sha256=source_sha256,
        validation_cases=expected_validation_cases,portability_migration=portable['portability_migration'])
    write_json(work/'context.json',context)
    print('VERIFIED: original epoch 0, exact CT/GT content, unchanged splits/groups; paths rebound for Colab',flush=True)
    return context


def environment():
    from importlib.metadata import version
    return dict(torch=str(torch.__version__),monai=version('monai'),einops=version('einops'),cuda_runtime=torch.version.cuda,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cuda_devices=torch.cuda.device_count() if torch.cuda.is_available() else 0)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


def durable_snapshot(trainer,context,path,signature,report,bn,plan):
    from tools.order_optimization_pilot import state_hash
    local=Path(context['work'])/(path.parent.name+'-save.pt')
    payload=dict(schema_version=SCHEMA,signature=signature,environment=environment(),report=report,
        input_binding=state_hash({key:context[key] for key in ('parent_sha256','source_sha256','hashes')}),
        model=trainer.model.state_dict(),optimizer=trainer.optimizer.state_dict(),scheduler=trainer.scheduler.state_dict(),
        scaler=trainer.scaler.state_dict(),rng=rng_state(),loader_generator=trainer.train_loader.generator.get_state(),
        initial_bn=bn,plan=plan,config=trainer.config,provenance=trainer.provenance,
        full_epoch_complete=False,native_trainer_resume_supported=False)
    torch.save(payload,local)
    pending=path.with_name(path.name+'.pending')
    shutil.copyfile(local,pending);os.replace(pending,path)
    write_json(path.parent/'arm-report.json',report)


def local_snapshot(context,path):
    import uuid
    local=Path(context['work'])/(path.parent.name+'-read-'+uuid.uuid4().hex+'.pt')
    shutil.copyfile(path,local)
    return local


def resume_training(trainer,path,signature,context):
    saved=torch.load(local_snapshot(context,path),map_location='cpu',mmap=True,weights_only=False)
    if saved.get('schema_version')!=SCHEMA or saved['signature']!=signature:
        raise ValueError('Snapshot protocol differs; use a new run folder for a changed experiment')
    if saved['environment']!=environment():
        raise ValueError('GPU/Torch environment changed; exact paired resume requires the previous environment')
    trainer.model.load_state_dict(saved['model'])
    trainer.optimizer.load_state_dict(saved['optimizer']);trainer.scheduler.load_state_dict(saved['scheduler'])
    trainer.scaler.load_state_dict(saved['scaler']);trainer.train_loader.generator.set_state(saved['loader_generator'])
    restore_rng(saved['rng'])
    return saved


def experiment_signature(context,trainer,plan,arm,total_batches,probe_ids):
    from tools.order_optimization_pilot import state_hash
    config=deepcopy(trainer.config)
    config['data']={key:('verified-path' if key in ('manifest','split_path','patient_map') else value)
                    for key,value in config['data'].items()}
    return state_hash(dict(parent_sha256=context['parent_sha256'],source_sha256=context['source_sha256'],
        data_hashes=context['hashes'],config=config,partitions=trainer.provenance['partitions'],
        plan=plan['mixed'],arm=arm,total_batches=total_batches,probe_ids=probe_ids,threshold=.5))


def run_arm(context,output,arm,*,total_batches=300,case_count=30,per_case=10,group_size=5,
            checkpoint_every=50,probe_every=50,max_seconds=None,max_batches_this_call=None,
            probe_ids=None):
    from tools.comprehensive_suite import setup,declared_plan,training_groups,train_one
    from tools.epoch1_frozen_bn_pilot import make_probes,evaluate_probes
    from merit_covid.data.pipeline.config import PreprocessConfig
    from merit_covid.data.pipeline.dataloader import collate_slabs
    from merit_covid.engine.training import protocol_fingerprint
    from tools.audit_bn_volume_pilot import bn_snapshot,bn_drift
    from tools.order_optimization_pilot import live_state_hashes,state_hash
    if arm not in ARMS:raise ValueError('Only the two predeclared alignment arms are supported')
    for value in (total_batches,checkpoint_every,probe_every):
        if type(value) is not int or value<1:raise ValueError('Batch budgets must be positive integers')
    if max_batches_this_call is not None and (type(max_batches_this_call) is not int or max_batches_this_call<1):
        raise ValueError('max_batches_this_call must be a positive integer')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pair_env=output/'pair-environment.json'
    if pair_env.exists() and json.loads(pair_env.read_text())!=environment():
        raise ValueError('Paired experiment environment changed; choose a new run folder')
    write_json(pair_env,environment())
    destination=output/arm;destination.mkdir(parents=True,exist_ok=True)
    trainer,datasets,parent=setup(context['parent'],context['hashes'],destination,{'name':arm})
    if arm=='alignment_off':
        trainer.criterion.alignment_weight=0.
        trainer.config['loss']['alignment_weight']=0.
        trainer.fingerprint=protocol_fingerprint(trainer.config,trainer.provenance)
    plan=declared_plan(datasets['train'],parent['loader_generator'],case_count,per_case,group_size)
    groups=training_groups(plan,{},total_batches,datasets['train'],parent['loader_generator'])
    probe_ids=probe_ids or sorted(c.case_id for c in datasets['validation'].cases)[:1]
    signature=experiment_signature(context,trainer,plan,arm,total_batches,probe_ids)
    progress=destination/'progress.pt'
    probes=make_probes(datasets['validation'].cases,PreprocessConfig(**trainer.config['preprocess']),probe_ids)
    initial_state=live_state_hashes(trainer)
    if progress.exists():
        saved=resume_training(trainer,progress,signature,context)
        report=deepcopy(saved['report']);bn=saved['initial_bn'];del saved
    else:
        bn=bn_snapshot(trainer.model)
        comparison_config=deepcopy(trainer.config)
        comparison_config['loss'].pop('alignment_weight')
        comparison_config['data']={key:('verified-path' if key in ('manifest','split_path','patient_map') else value)
                                  for key,value in comparison_config['data'].items()}
        report=dict(arm=arm,signature=signature,status='training',requested_batches=total_batches,
            comparison_binding=state_hash(dict(config=comparison_config,parent=context['parent_sha256'],
                source=context['source_sha256'],data=context['hashes'],plan=plan['mixed'],
                total_batches=total_batches,probe_ids=probe_ids)),
            processed_batches=0,optimizer_steps=0,amp_skipped_steps=0,train=[],
            probes=[evaluate_probes(trainer,probes,0)],bn_checks=[],initial_state_sha256=initial_state,
            alignment_weight=trainer.criterion.alignment_weight,full_epoch_complete=False,
            full_validation_performed=False,environment=environment(),plan=plan['mixed'][:total_batches])
        write_json(destination/'arm-report.json',report)
    done=report['processed_batches'];this_call=0;started=time.monotonic()
    deadline=started+max_seconds if max_seconds is not None else float('inf')
    if done>=total_batches:
        print(arm,'training already complete; saved model ready for validation',flush=True)
        return report
    try:
        offset=0
        for indices in groups:
            end=offset+len(indices)
            if end<=done:offset=end;continue
            # Stage each group once in volume order; resume skips completed slabs.
            remaining=indices[max(0,done-offset):]
            cache={index:collate_slabs([datasets['train'][index]]) for index in sorted(remaining)}
            datasets['train']._cached=None;datasets['train']._cached_index=None
            for index in remaining:
                if time.monotonic()>=deadline:
                    durable_snapshot(trainer,context,progress,signature,report,bn,plan)
                    write_json(destination/'run-status.json',dict(status='training_paused',processed_batches=done))
                    return report
                row=train_one(trainer,cache[index],{})
                done+=1;this_call+=1;row['batch']=done;report['train'].append(row)
                if [row['case'],row['master']]!=plan['mixed'][done-1]:raise AssertionError('Sampling plan mismatch')
                report.update(processed_batches=done,optimizer_steps=done)
                if done%10==0:print('COLAB TRAIN',arm,done,total_batches,row['optimized_total'],flush=True)
                stopping=max_batches_this_call is not None and this_call>=max_batches_this_call
                if done%probe_every==0 or done==total_batches:
                    report['probes'].append(evaluate_probes(trainer,probes,done))
                if done%checkpoint_every==0 or done==total_batches or stopping:
                    drift=bn_drift(bn,bn_snapshot(trainer.model))
                    if drift['changed_buffers']:raise AssertionError('Frozen BN buffers changed')
                    report['bn_checks'].append(dict(batch=done,**drift))
                    report['status']='training_complete' if done==total_batches else 'training'
                    if done==total_batches:
                        report['model_sha256']=state_hash(trainer.model.state_dict())
                    durable_snapshot(trainer,context,progress,signature,report,bn,plan)
                    print('DRIVE CHECKPOINT SAVED',arm,done,flush=True)
                if stopping and done<total_batches:return report
            offset=end;del cache;gc.collect()
        report.update(status='training_complete')
        write_json(destination/'arm-report.json',report)
        write_json(destination/'run-status.json',dict(status='training_complete',processed_batches=done))
        return report
    except BaseException as error:
        # The previous completed atomic snapshot remains usable, including on OOM.
        write_json(destination/'run-status.json',dict(status='training_interrupted',error=str(error),
            last_saved_checkpoint=str(progress),processed_in_memory=done))
        raise


def validate_arm(context,output,arm,*,max_seconds=None,max_cases_this_call=None):
    from tools.comprehensive_suite import setup
    from merit_covid.engine.training import evaluate_model
    from merit_covid.data.pipeline.metrics import aggregate_results
    from tools.order_optimization_pilot import state_hash
    destination=Path(output)/arm
    progress=destination/'progress.pt'
    if not progress.exists():raise FileNotFoundError('Run training for this arm first')
    saved=torch.load(local_snapshot(context,progress),map_location='cpu',mmap=True,weights_only=False)
    if saved.get('schema_version')!=SCHEMA:raise ValueError('Unsupported experiment checkpoint')
    if saved.get('input_binding')!=state_hash({key:context[key] for key in ('parent_sha256','source_sha256','hashes')}):
        raise ValueError('Validation checkpoint source or input identity changed')
    report=saved['report']
    if report['processed_batches']!=report['requested_batches']:
        print(arm,'training incomplete; validation deferred',flush=True);return dict(status='training_incomplete')
    if saved['environment']!=environment():raise ValueError('Validation environment changed; paired comparison requires the same GPU/Torch')
    trainer,datasets,parent=setup(context['parent'],context['hashes'],destination,{'name':arm})
    trainer.model.load_state_dict(saved['model']);trainer.model.eval()
    signature=saved['signature'];before=state_hash(trainer.model.state_dict());del saved
    partial_path=destination/'validation-partial.json'
    partial=json.loads(partial_path.read_text()) if partial_path.exists() else dict(signature=signature,model_sha256=before,per_case=[],threshold=.5)
    if partial['signature']!=signature or partial['model_sha256']!=before:raise ValueError('Partial predictions belong to another model/protocol')
    baseline=json.loads(Path(context['history']).read_text().strip())['validation']
    old={r['case_id']:r for r in baseline['per_case']}
    if len({r['case_id'] for r in partial['per_case']})!=len(partial['per_case']):raise ValueError('Duplicate partial validation cases')
    expected={c.case_id for c in trainer.validation_cases}
    if not {r['case_id'] for r in partial['per_case']}.issubset(expected):raise ValueError('Unexpected partial case ID')
    done={r['case_id'] for r in partial['per_case']};started=time.monotonic();new_cases=0
    for case in trainer.validation_cases:
        if case.case_id in done:continue
        if (max_seconds is not None and time.monotonic()-started>=max_seconds) or (max_cases_this_call is not None and new_cases>=max_cases_this_call):
            return dict(status='validation_paused',completed_cases=len(done),requested_cases=len(expected))
        print('COLAB NATIVE VALIDATION',arm,len(done)+1,len(expected),case.case_id,flush=True)
        count=0
        def heartbeat(module,args):
            nonlocal count
            count+=len(args[0])
            if count==1 or count%50==0:print(case.case_id,'slices',count,flush=True)
        hook=trainer.model.register_forward_pre_hook(heartbeat)
        try:
            result=evaluate_model(trainer.model,[case],trainer.preprocess,threshold=.5,
                include_hd95=trainer.settings['include_hd95'],batch_size=1,num_workers=0,
                identity=trainer.provenance.get('identity'))
        finally:hook.remove()
        row=result['per_case'][0]
        if row['ground_truth_foreground_voxels']!=old[case.case_id]['ground_truth_foreground_voxels']:
            raise AssertionError('Ground-truth voxel count differs from epoch-0 baseline')
        partial['per_case'].append(row);done.add(case.case_id);new_cases+=1
        partial.update(completed_cases=len(done),requested_cases=len(expected))
        write_json(partial_path,partial)
        print('DRIVE CASE SAVED',case.case_id,row['dice'],flush=True);gc.collect()
    if state_hash(trainer.model.state_dict())!=before:raise AssertionError('Validation changed the saved model')
    complete=aggregate_results(partial['per_case'],trainer.provenance.get('identity'))
    complete.update(threshold=.5,evaluation_grid='full original NIfTI XYZ',signature=signature,
        full_validation_performed=True,validation_cases=len(expected),model_sha256=before)
    write_json(destination/'validation-native.json',complete)
    report.update(full_validation_performed=True,status='complete',native_macro=complete['macro'],
        validation_cases=len(expected),fp=sum(r['fp'] for r in partial['per_case']),
        empty_predictions=sum(r['prediction_foreground_voxels']==0 for r in partial['per_case']))
    write_json(destination/'arm-report.json',report);write_json(destination/'run-status.json',dict(status='complete',validation_cases=len(expected)))
    return complete


def compare(output):
    output=Path(output);rows=[]
    for arm in ARMS:
        path=output/arm/'validation-native.json'
        if not path.exists():
            print(arm,'not yet fully validated');continue
        v=json.loads(path.read_text())
        report=json.loads((output/arm/'arm-report.json').read_text())
        rows.append(dict(arm=arm,cases=v['validation_cases'],dice=v['macro']['dice']['mean'],
            precision=v['macro']['precision']['mean'],recall=v['macro']['recall']['mean'],
            fp=sum(r['fp'] for r in v['per_case']),empty=sum(r['prediction_foreground_voxels']==0 for r in v['per_case']),
            steps=report['processed_batches']))
    comparison=dict(status='complete' if len(rows)==2 else 'partial',arms=rows,
        interpretation='Alignment-only ablation on development validation, not proof of a unique cause or unbiased test performance')
    if len(rows)==2:
        a=json.loads((output/ARMS[0]/'arm-report.json').read_text());b=json.loads((output/ARMS[1]/'arm-report.json').read_text())
        if a.get('comparison_binding')!=b.get('comparison_binding') or a['initial_state_sha256']!=b['initial_state_sha256'] or [[r['case'],r['master']] for r in a['train']]!=[[r['case'],r['master']] for r in b['train']]:
            raise AssertionError('The two arms were not matched at restart/sample order')
        comparison['matched_initial_state_and_samples']=True
    write_json(output/'comparison.json',comparison)
    for row in rows:print(json.dumps(row))
    return comparison
