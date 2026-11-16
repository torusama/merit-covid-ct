"""Executed inside the frozen epoch-1 source, not the working-tree trainer."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import Trainer,load_experiment,split_datasets,protocol_fingerprint,load_checkpoint_model,validate_experiment
from merit_covid.data.pipeline.dataloader import make_dataloader
from scripts.kaggle.fast_epoch1 import migrate_execution, run
from scripts.train_colab import train
from merit_covid.engine.training import TRAIN_DEFAULTS

pytestmark=pytest.mark.skipif('mer_recomputation' not in TRAIN_DEFAULTS,
    reason='Run this suite inside outputs/kaggle_fast_epoch1/test_source, the frozen deployment source')


def new_trainer(config,output):
    datasets,provenance=split_datasets(config,0)
    provenance['train_order']='seeded case shuffle, then seeded master shuffle within case; every master once'
    loader=make_dataloader(datasets['train'],batch_size=1,shuffle=True,shuffle_mode='volume',seed=config['training']['seed'],num_workers=0)
    return Trainer(config,loader,datasets['validation'].cases,provenance,output)


def test_epoch_boundary_migration_save_reload_and_one_epoch(tmp_path):
    config=synthetic_experiment(tmp_path/'data',device='cpu')
    config['training'].update(activation_checkpointing=True,batch_size=1,shuffle_mode='volume')
    path=tmp_path/'original.yaml';path.write_text(yaml.safe_dump(config))
    old,history=train(path,0,tmp_path/'original',epochs=1)
    assert len(history)==1
    cp=torch.load(tmp_path/'original/last.pt',map_location='cpu',weights_only=False)
    target=deepcopy(cp['config']);target['training'].update(epochs=2,deterministic=False,factorized_mer=True,mer_recomputation=False)
    target['model']['query_chunk_size']=4096;target=validate_experiment(target)
    trainer=new_trainer(target,tmp_path/'fast')
    migrated=migrate_execution(cp,target,trainer.provenance)
    for key in ('model','optimizer','scheduler','scaler','rng','loader_generator'):assert migrated[key] is cp[key]
    trainer.resume(migrated)
    assert trainer.start_epoch==1 and trainer.scheduler.state_dict()==cp['scheduler'] and trainer.scaler.state_dict()==cp['scaler']
    assert trainer.model.mer._mer_recomputation is False
    states=cp['optimizer']['param_groups'][0]['params']
    for p,stateid in zip(trainer.model.parameters(),states):
        for key in ('exp_avg','exp_avg_sq','step'):torch.testing.assert_close(trainer.optimizer.state[p][key],cp['optimizer']['state'][stateid][key],rtol=0,atol=0)
    rows=trainer.fit(resume=migrated,checkpoint_every=1,max_seconds=None)
    assert len(rows)==1 and rows[0]['epoch']==1
    reloaded,saved=load_checkpoint_model(tmp_path/'fast/last.pt')
    assert saved['epoch']==1 and saved['config']['training']['epochs']==2
    assert saved['config']['training']['mer_recomputation'] is False
    assert saved['config']['training']['factorized_mer'] is True
    assert saved['execution_migration']['from_epoch']==0
    assert reloaded.mer._mer_recomputation is False
    assert not (tmp_path/'fast/in_progress.pt').exists()
    for name,value in trainer.model.state_dict().items():torch.testing.assert_close(value,reloaded.state_dict()[name],rtol=0,atol=0)


def test_continuation_creates_checkpoint_directory_before_copy(tmp_path):
    config=synthetic_experiment(tmp_path/'data',device='cpu')
    config['training'].update(activation_checkpointing=True,batch_size=1,shuffle_mode='volume')
    path=tmp_path/'original.yaml';path.write_text(yaml.safe_dump(config))
    train(path,0,tmp_path/'original',epochs=1)
    datasets,_=split_datasets(load_experiment(path),0)
    hashes={case.case_id:{key:hashlib.sha256(gzip.decompress(Path(source).read_bytes())).hexdigest()
        for key,source in [('image',case.image),('mask',case.mask)]}
        for dataset in datasets.values() for case in dataset.cases}
    output=tmp_path/'fresh-kaggle-working'/'MERIT'/'fold-00'
    assert not output.exists()

    run(tmp_path/'original/last.pt',tmp_path/'original/history.jsonl',
        tmp_path/'original/best.pt',path,hashes,output)

    assert (output/'checkpoints/best.pt').is_file()
    rows=[json.loads(line) for line in (output/'checkpoints/history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in rows]==[0,1]
    assert rows[0]==json.loads((tmp_path/'original/history.jsonl').read_text())
    checkpoint=torch.load(output/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==1
    assert json.loads((output/'run-status.json').read_text())['status']=='epoch_complete'


def fixture():
    config=validate_experiment(dict(data={'manifest':'/tmp/a.json','split_path':'/tmp/b.json'},
        preprocess={'slab_size':11,'size_hw':[32,32]},loss={},
        model={'img_size':[11,32,32]},training={'epochs':1,'activation_checkpointing':True}))
    provenance={'source_identity':{'cases':[{'image':{'mtime_ns':1},'mask':{'mtime_ns':1}}]}}
    cp=dict(schema_version=2,epoch=0,config=config,provenance=provenance,
        protocol_fingerprint=protocol_fingerprint(config,provenance))
    cp.update({key:object() for key in ('model','optimizer','scheduler','scaler','rng','loader_generator')})
    target=deepcopy(config);target['training'].update(epochs=2,deterministic=False,factorized_mer=True,mer_recomputation=False);target['model']['query_chunk_size']=4096
    return cp,target,provenance


@pytest.mark.parametrize('change',['lr','loss','threshold','sampling','mid_epoch','wrong_epoch'])
def test_reject_unapproved_migration(change):
    cp,target,provenance=fixture()
    if change=='lr':target['training']['learning_rate']*=2
    elif change=='loss':target['loss']['presence_weight']=.5
    elif change=='threshold':target['training']['threshold']=.4
    elif change=='sampling':target['training']['samples_per_volume']=10
    elif change=='mid_epoch':cp['in_progress']={'batches':10}
    else:cp['epoch']=1
    with pytest.raises(ValueError):migrate_execution(cp,target,provenance)


def test_only_timestamps_may_change_in_source_binding():
    cp,target,provenance=fixture();current=deepcopy(provenance)
    current['source_identity']['cases'][0]['image']['mtime_ns']=2
    migrated=migrate_execution(cp,target,current);assert migrated['model'] is cp['model']
    current['source_identity']['cases'][0]['image']['path']='other'
    with pytest.raises(ValueError):migrate_execution(cp,target,current)


def test_default_execution_fields_keep_legacy_fingerprint():
    cp,_,provenance=fixture();legacy=deepcopy(cp['config'])
    legacy['training'].pop('factorized_mer');legacy['training'].pop('mer_recomputation')
    assert protocol_fingerprint(legacy,provenance)==cp['protocol_fingerprint']
