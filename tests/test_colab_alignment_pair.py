"""Execute in the frozen notebook test_source; no Colab/GPU credentials used."""
from copy import deepcopy
import ast
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import TRAIN_DEFAULTS,load_experiment,split_datasets
from scripts.train_colab import train
from tools.colab_alignment_pair import (
    bind_portable_checkpoint,file_hash,prepare,run_arm,validate_arm,compare,stage_dataset,find_dataset_root,
)
from tools.order_optimization_pilot import state_hash

pytestmark=pytest.mark.skipif('mer_recomputation' not in TRAIN_DEFAULTS,
    reason='Run against outputs/colab_alignment_pair/test_source')


@pytest.fixture(scope='module')
def context(tmp_path_factory):
    root=tmp_path_factory.mktemp('colab-alignment')
    config=synthetic_experiment(root/'original-data',device='cpu')
    for path in (root/'original-data').glob('*.nii.gz'):
        image=nib.load(path);array=image.get_fdata(dtype=np.float32)
        values=np.pad(array,((0,32-array.shape[0]),(0,32-array.shape[1]),(0,0)))
        nib.save(nib.Nifti1Image(values,image.affine,image.header),path)
    config['preprocess']['size_hw']=None
    config['training'].update(activation_checkpointing=True,batch_size=1,shuffle_mode='volume')
    path=root/'experiment.yaml';path.write_text(yaml.safe_dump(config))
    train(path,0,root/'parent',epochs=1)
    parent=root/'parent/last.pt';history=root/'parent/history.jsonl'
    datasets,_=split_datasets(load_experiment(path),0)
    hashes={case.case_id:{key:hashlib.sha256(gzip.decompress(Path(source).read_bytes())).hexdigest()
        for key,source in [('image',case.image),('mask',case.mask)]}
        for ds in datasets.values() for case in ds.cases}
    dataset=root/'relocated-dataset/Train';dataset.mkdir(parents=True)
    for ds in datasets.values():
        for case in ds.cases:
            shutil.copy2(case.image,dataset/(case.case_id+'_ct.nii.gz'))
            shutil.copy2(case.mask,dataset/(case.case_id+'_seg.nii.gz'))
    project=root/'new-project'
    manifest=project/'data/covid19_20/cases.json';manifest.parent.mkdir(parents=True)
    shutil.copy2(config['data']['manifest'],manifest)
    split=project/'splits/covid19_20_5fold.json';split.parent.mkdir(parents=True)
    shutil.copy2(config['data']['split_path'],split)
    result=prepare(project,root/'relocated-dataset',parent,history,hashes,root/'scratch',
        parent_sha256=file_hash(parent),source_sha256='a'*64,
        expected_cases=10,expected_validation_cases=len(datasets['validation'].cases))
    result['test_root']=str(root)
    result['test_original_parent']=str(parent)
    result['test_probes']=[case.case_id for case in datasets['validation'].cases]
    return result


def test_portable_paths_preserve_training_state_and_reject_nonpath_changes(context):
    original=torch.load(context['test_original_parent'],map_location='cpu',weights_only=False)
    portable=torch.load(context['parent'],map_location='cpu',weights_only=False)
    assert original['config']['data']!=portable['config']['data']
    for key in ('model','optimizer','scheduler','scaler','rng','loader_generator'):
        assert state_hash(original[key])==state_hash(portable[key])
    assert original['provenance']['partitions']==portable['provenance']['partitions']
    changed=deepcopy(portable['config']);changed['loss']['alignment_weight']=0.
    with pytest.raises(ValueError,match='data paths only'):
        bind_portable_checkpoint(original,changed,portable['provenance'])
    changed_provenance=deepcopy(portable['provenance'])
    changed_provenance['all_group_ids'].append('unapproved-group')
    with pytest.raises(ValueError,match='split/group identity'):
        bind_portable_checkpoint(original,portable['config'],changed_provenance)


def test_two_gpu_parent_can_bind_one_stream_without_mutating_original(context):
    original=torch.load(context['test_original_parent'],map_location='cpu',weights_only=False)
    portable=torch.load(context['parent'],map_location='cpu',weights_only=False)
    original['rng']['cuda']=[torch.tensor([1,2],dtype=torch.uint8),torch.tensor([3,4],dtype=torch.uint8)]
    rebound=bind_portable_checkpoint(original,portable['config'],portable['provenance'],cuda_devices=1)
    assert len(original['rng']['cuda'])==2 and len(rebound['rng']['cuda'])==1
    assert torch.equal(rebound['rng']['cuda'][0],original['rng']['cuda'][0])
    with pytest.raises(ValueError,match='missing CUDA RNG'):
        bind_portable_checkpoint(original,portable['config'],portable['provenance'],cuda_devices=3)


@pytest.mark.parametrize('arm',['reference_alignment','alignment_off'])
def test_midgroup_resume_matches_continuous_model_optimizer_and_rng(context,tmp_path,arm):
    options=dict(total_batches=4,case_count=4,per_case=2,group_size=2,
        checkpoint_every=2,probe_every=2,probe_ids=context['test_probes'])
    parent_hash=file_hash(context['test_original_parent'])
    complete=run_arm(context,tmp_path/'full',arm,**options)
    gc_collect()
    paused=run_arm(context,tmp_path/'resume',arm,max_batches_this_call=1,**options)
    assert paused['processed_batches']==1
    gc_collect()
    resumed=run_arm(context,tmp_path/'resume',arm,**options)
    assert complete['train']==resumed['train']
    assert complete['probes']==resumed['probes']
    assert resumed['processed_batches']==4
    a=torch.load(tmp_path/'full'/arm/'progress.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'resume'/arm/'progress.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','scaler','rng','loader_generator'):
        assert state_hash(a[key])==state_hash(b[key]),key
    assert all(check['changed_buffers']==0 for check in resumed['bn_checks'])
    assert complete['alignment_weight']==(0. if arm=='alignment_off' else .05)
    assert file_hash(context['test_original_parent'])==parent_hash
    for key,state in torch.load(context['parent'],map_location='cpu',weights_only=False)['optimizer']['state'].items():
        assert a['optimizer']['state'][key]['step'].item()==state['step'].item()+4


def gc_collect():
    import gc
    gc.collect()


def test_validation_resume_skips_saved_cases_and_pair_changes_alignment_only(context,tmp_path,monkeypatch):
    from merit_covid.engine import training
    options=dict(total_batches=4,case_count=4,per_case=2,group_size=2,
        checkpoint_every=2,probe_every=2,probe_ids=context['test_probes'])
    for arm in ('reference_alignment','alignment_off'):
        run_arm(context,tmp_path,arm,**options)
        gc_collect()
    a=torch.load(tmp_path/'reference_alignment/progress.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'alignment_off/progress.pt',map_location='cpu',weights_only=False)
    expected=deepcopy(a['config']);expected['loss']['alignment_weight']=0.
    assert b['config']==expected
    assert a['report']['initial_state_sha256']==b['report']['initial_state_sha256']
    for arm in ('reference_alignment','alignment_off'):
        first=validate_arm(context,tmp_path,arm,max_cases_this_call=1)
        assert first['status']=='validation_paused'
        partial=json.loads((tmp_path/arm/'validation-partial.json').read_text())
        seen={row['case_id'] for row in partial['per_case']}
        original_evaluate=training.evaluate_model
        def checking(model,cases,*args,**kwargs):
            assert not any(case.case_id in seen for case in cases),'Already saved case was reevaluated'
            return original_evaluate(model,cases,*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(training,'evaluate_model',checking)
            complete=validate_arm(context,tmp_path,arm)
        assert complete['full_validation_performed'] and complete['validation_cases']==context['validation_cases']
        assert complete['threshold']==.5
        gc_collect()
    comparison=compare(tmp_path)
    assert comparison['status']=='complete' and comparison['matched_initial_state_and_samples']


def test_changed_protocol_does_not_resume(context,tmp_path):
    options=dict(total_batches=4,case_count=4,per_case=2,group_size=2,
        checkpoint_every=2,probe_every=2,probe_ids=context['test_probes'])
    run_arm(context,tmp_path,'reference_alignment',max_batches_this_call=1,**options)
    gc_collect()
    changed=deepcopy(context);changed['source_sha256']='b'*64
    with pytest.raises(ValueError,match='Snapshot protocol differs'):
        run_arm(changed,tmp_path,'reference_alignment',**options)
    gc_collect()
    with pytest.raises(ValueError,match='input identity changed'):
        validate_arm(changed,tmp_path,'reference_alignment')


def test_unfinished_training_never_claims_full_native_validation(context,tmp_path):
    run_arm(context,tmp_path,'reference_alignment',total_batches=4,case_count=4,per_case=2,
        group_size=2,checkpoint_every=2,probe_every=2,max_batches_this_call=1,probe_ids=context['test_probes'])
    gc_collect()
    result=validate_arm(context,tmp_path,'reference_alignment')
    assert result['status']=='training_incomplete'
    assert compare(tmp_path)['status']=='partial'
    assert not (tmp_path/'reference_alignment/validation-native.json').exists()


def test_zip_staging_rejects_escape_and_accepts_safe_archive(tmp_path):
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as archive:
        archive.writestr('../escape.txt','bad')
    with pytest.raises(ValueError,match='escaping path'):
        stage_dataset(bad,tmp_path/'bad-target')
    assert not (tmp_path/'escape.txt').exists()
    good=tmp_path/'good.zip'
    with zipfile.ZipFile(good,'w') as archive:
        archive.writestr('Train/sample.nii','bytes')
    staged=stage_dataset(good,tmp_path/'good-target')
    assert (staged/'Train/sample.nii').read_text()=='bytes'


def test_flat_inputs_stage_nifti_without_copying_checkpoint(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'volume-covid19-A-0003_ct.nii.gz').write_bytes(b'ct')
    (inputs/'volume-covid19-A-0003_seg.nii.gz').write_bytes(b'gt')
    epoch0=inputs/'epoch0';epoch0.mkdir()
    (epoch0/'last.pt').write_bytes(b'checkpoint')
    (epoch0/'history.jsonl').write_text('{}')
    staged=stage_dataset(inputs,tmp_path/'staged')
    assert (staged/'volume-covid19-A-0003_ct.nii.gz').read_bytes()==b'ct'
    assert (staged/'volume-covid19-A-0003_seg.nii.gz').read_bytes()==b'gt'
    assert not list(staged.rglob('*.pt')) and not list(staged.rglob('*.jsonl'))


@pytest.mark.parametrize('layout',['flat','arbitrary_nested','separate_ct_gt','kaggle_nested'])
def test_dataset_discovery_does_not_require_train_folder_name(tmp_path,layout):
    ids=['volume-covid19-A-0003','volume-covid19-A-0196_0']
    for case_id in ids:
        for key in ('ct','seg'):
            stem=f'{case_id}_{key}'
            if layout=='flat':
                path=tmp_path/(stem+'.nii.gz')
            elif layout=='arbitrary_nested':
                path=tmp_path/'my_upload'/'data_files'/(stem+'.nii.gz')
            elif layout=='separate_ct_gt':
                path=tmp_path/('images' if key=='ct' else 'masks')/(stem+'.nii')
            else:
                path=tmp_path/'download'/(stem+'.nii')/(case_id+'.nii')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'fixture')
    source,paths=find_dataset_root(tmp_path,ids)
    assert source==tmp_path
    assert set(paths)==set(ids)
    assert all(p.is_file() for pair in paths.values() for p in pair.values())


def test_dataset_discovery_identifies_missing_gt_and_duplicate_ct(tmp_path):
    case='volume-covid19-A-0003'
    (tmp_path/(case+'_ct.nii.gz')).write_bytes(b'ct')
    with pytest.raises(FileNotFoundError) as error:
        find_dataset_root(tmp_path,[case])
    assert 'CT 1/1, GT 0/1' in str(error.value)
    assert case+'_seg' in str(error.value)
    (tmp_path/(case+'_seg.nii.gz')).write_bytes(b'gt')
    duplicate=tmp_path/'another-copy';duplicate.mkdir()
    (duplicate/(case+'_ct.nii.gz')).write_bytes(b'ct')
    with pytest.raises(ValueError,match='Có nhiều bản CT/seg'):
        find_dataset_root(tmp_path,[case])


def test_staging_refreshes_nested_upload_even_when_root_mtime_is_unchanged(tmp_path):
    import os
    inputs=tmp_path/'inputs';data=inputs/'other_folder';data.mkdir(parents=True)
    ct=data/'volume-covid19-A-0003_ct.nii.gz';ct.write_bytes(b'ct')
    stat=inputs.stat()
    staged=stage_dataset(inputs,tmp_path/'cache')
    assert (staged/'other_folder'/ct.name).is_file()
    gt=data/'volume-covid19-A-0003_seg.nii.gz';gt.write_bytes(b'gt')
    os.utime(inputs,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    stage_dataset(inputs,tmp_path/'cache')
    assert (staged/'other_folder'/gt.name).read_bytes()==b'gt'


def notebook_file():
    return next(parent/'notebooks/colab_alignment_pair.ipynb'
                for parent in Path(__file__).resolve().parents
                if (parent/'notebooks/colab_alignment_pair.ipynb').exists())


def input_resolver():
    value=json.loads(notebook_file().read_text(encoding='utf-8'))
    code=[cell for cell in value['cells'] if cell['cell_type']=='code']
    tree=ast.parse(''.join(code[2]['source']))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                  and node.name=='resolve_notebook_inputs')
    namespace={'Path':Path}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<input-cell>','exec'),namespace)
    return namespace['resolve_notebook_inputs']


def test_notebook_resolves_confirmed_my_drive_tree(tmp_path):
    value=json.loads(notebook_file().read_text(encoding='utf-8'))
    code=[cell for cell in value['cells'] if cell['cell_type']=='code']
    config={}
    exec(''.join(code[0]['source']),config)
    root='/content/drive/MyDrive/MERIT_alignment_test'
    assert config['DRIVE_ROOT']==root
    assert config['DATASET_SOURCE']==root+'/inputs/Train'
    assert config['EPOCH0_CHECKPOINT']==root+'/inputs/epoch0/last.pt'
    assert config['EPOCH0_HISTORY']==root+'/inputs/epoch0/history.jsonl'
    assert config['FORCE_REMOUNT_DRIVE'] is False
    inputs=tmp_path/'MyDrive/MERIT_alignment_test/inputs'
    train=inputs/'Train';train.mkdir(parents=True)
    (train/'volume-covid19-A-0003_ct.nii.gz').write_bytes(b'ct')
    (train/'volume-covid19-A-0003_seg.nii.gz').write_bytes(b'gt')
    epoch0=inputs/'epoch0';epoch0.mkdir()
    (epoch0/'last.pt').write_bytes(b'parent')
    (epoch0/'history.jsonl').write_text('{}')
    result=input_resolver()(inputs,train,epoch0/'last.pt',epoch0/'history.jsonl')
    assert result==(str(train),str(epoch0/'last.pt'),str(epoch0/'history.jsonl'))


def test_notebook_missing_inputs_lists_mounted_drive_folders(tmp_path):
    my_drive=tmp_path/'MyDrive';my_drive.mkdir()
    (my_drive/'other-account-folder').mkdir()
    inputs=my_drive/'MERIT_alignment_test/inputs'
    with pytest.raises(FileNotFoundError) as error:
        input_resolver()(inputs,inputs/'Train',inputs/'epoch0/last.pt',inputs/'epoch0/history.jsonl')
    message=str(error.value)
    assert str(inputs) in message
    assert 'other-account-folder' in message
    assert 'Google account' in message


@pytest.mark.parametrize('nested',[True,False])
def test_notebook_resolves_observed_flat_data_and_checkpoint_locations(tmp_path,nested):
    inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'volume-covid19-A-0003_ct.nii.gz').write_bytes(b'ct')
    checkpoint_dir=inputs/'elsewhere/epoch0' if nested else inputs
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    (checkpoint_dir/'last.pt').write_bytes(b'parent')
    (checkpoint_dir/'history.jsonl').write_text('{}')
    dataset,parent,history=input_resolver()(inputs,inputs/'Train',
        inputs/'epoch0/last.pt',inputs/'epoch0/history.jsonl')
    assert dataset==str(inputs)
    assert parent==str(checkpoint_dir/'last.pt')
    assert history==str(checkpoint_dir/'history.jsonl')


def test_notebook_reports_missing_checkpoint_and_rejects_ambiguous_files(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    resolve=input_resolver()
    with pytest.raises(FileNotFoundError) as error:
        resolve(inputs,inputs,inputs/'epoch0/last.pt',inputs/'epoch0/history.jsonl')
    assert 'last.pt' in str(error.value) and 'history.jsonl' in str(error.value)
    assert str(inputs/'epoch0/last.pt') in str(error.value)
    assert str(inputs/'epoch0/history.jsonl') in str(error.value)
    assert str(inputs) in str(error.value)
    for name in ('one','two'):
        folder=inputs/name;folder.mkdir()
        (folder/'last.pt').write_bytes(b'parent')
    (inputs/'history.jsonl').write_text('{}')
    with pytest.raises(FileNotFoundError,match='Có nhiều last.pt'):
        resolve(inputs,inputs,inputs/'missing.pt',inputs/'missing-history')


def test_notebook_finds_nested_dataset_and_single_zip_when_old_path_is_missing(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'last.pt').write_bytes(b'parent')
    (inputs/'history.jsonl').write_text('{}')
    folder=inputs/'not-named-Train'/'ct';folder.mkdir(parents=True)
    (folder/'volume-covid19-A-0003_ct.nii.gz').write_bytes(b'ct')
    resolve=input_resolver()
    result=resolve(inputs,inputs/'old-missing-path',inputs/'last.pt',inputs/'history.jsonl')
    assert result[0]==str(inputs)
    (folder/'volume-covid19-A-0003_ct.nii.gz').unlink()
    archive=inputs/'my-dataset.zip';archive.write_bytes(b'zip-placeholder')
    result=resolve(inputs,inputs/'old-missing-path',inputs/'last.pt',inputs/'history.jsonl')
    assert result[0]==str(archive)


def test_notebook_is_unexecuted_valid_json_and_python_cells():
    # Locate the repository notebook when run in either root or frozen output.
    here=Path(__file__).resolve()
    candidates=[parent/'notebooks/colab_alignment_pair.ipynb' for parent in here.parents]
    notebook=next(path for path in candidates if path.exists())
    value=json.loads(notebook.read_text(encoding='utf-8'))
    import nbformat
    nbformat.validate(nbformat.from_dict(value))
    code=[cell for cell in value['cells'] if cell['cell_type']=='code']
    assert len(code)==12
    for index,cell in enumerate(code,1):
        assert ''.join(cell['source']).startswith(f'#@title Cell {index} ')
    assert code[3]['metadata']['cellView']=='form'
    assert "'reference_alignment'" in ''.join(code[7]['source'])
    assert "'alignment_off'" in ''.join(code[8]['source'])
    assert 'run_arm(' in ''.join(code[7]['source']) and 'run_arm(' in ''.join(code[8]['source'])
    assert 'validate_arm(' in ''.join(code[9]['source']) and 'validate_arm(' in ''.join(code[10]['source'])
    for cell in value['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
            assert cell['execution_count'] is None and cell['outputs']==[]
    export=notebook.with_name('colab_alignment_pair_cells.py').read_text(encoding='utf-8')
    ast.parse(export)
    assert sum(line=='# %%' for line in export.splitlines())==12
