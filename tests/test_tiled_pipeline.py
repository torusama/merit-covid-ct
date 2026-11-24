"""B3B software fixtures use the actual exporter schema, never real training."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import time
import types

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from merit_covid.data.pipeline import tiled as t
from merit_covid.data.pipeline.cases import load_cases
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataloader import (make_dataloader, collate_slabs, VolumeShuffleSampler,
                                                VolumeBalancedSampler, resume_loader_iterator)
from merit_covid.data.pipeline.splits import create_splits
from merit_covid.engine.prediction import VolumePrediction
from merit_covid.engine.tiled_inference import TileProbabilityReconstructor
from merit_covid.engine.training import (load_experiment, validate_experiment, split_datasets, Trainer,
                                        evaluate_model, protocol_fingerprint, load_checkpoint_model)
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITOutput

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def geom(image):
    q, qc = image.get_qform(coded=True); s, sc = image.get_sform(coded=True)
    return dict(shape_xyz=list(image.shape), affine=image.affine.tolist(),
        orientation="".join(nib.aff2axcodes(image.affine)), spacing=nib.affines.voxel_sizes(image.affine).tolist(),
        spatial_unit=image.header.get_xyzt_units()[0], qform_code=int(qc), sform_code=int(sc),
        qform=q.tolist() if q is not None else None, sform=s.tolist() if s is not None else None)


def fixture_cohort(root, n=10, metadata_only=False):
    """Small real NIfTIs; the 199-count test uses EMPTY placeholder files only."""
    root.mkdir(parents=True)
    meta = root / 'source_metadata'; meta.mkdir()
    source_records = [dict(id=f'synthetic-{i:03d}', image=f'synthetic-{i:03d}_ct.nii.gz',
                           mask=f'synthetic-{i:03d}_seg.nii', group_id=f'group-{i:03d}') for i in range(n)]
    dump(meta/'cases.json', source_records)
    cases, identity = load_cases(meta/'cases.json', check_files=False)
    split = create_splits(cases, identity)
    dump(meta/'splits_covid19_20_5fold.json', split)
    partitions = {cid: role for role in ('train','validation','test') for cid in split['folds'][0][role]['case_ids']}
    config = yaml.safe_load((ROOT/'configs/merit_tiling128_v3_fold0.yaml').read_text(encoding='utf-8'))
    (meta/'merit_precomputed_v3.yaml').write_text(yaml.safe_dump({'preprocess': config['preprocess']}))
    depths = ([69]*173 + [68]*26) if n == 199 else [1,3,13] + [3]*(n-3)
    full_by_id = {}; rows = []; source_files = []
    for i, source in enumerate(source_records):
        cid = source['id']; depth = depths[i]
        affine = np.diag([-.7, -.8, 2.5, 1.]); affine[:3,3] = [31,-72,5]
        affine[0,1] = .025
        if metadata_only:
            # Geometry needs only headers; no large full voxel allocation.
            full_img = nib.Nifti1Image(np.zeros((1,1,1), np.float32), affine)
            full_img.header.set_data_shape((512,512,depth))
        else:
            full = np.full((512,512,depth), -1.25, np.float32)
            for z in range(depth):
                full[127:129,127:129,z] = 2.5
                full[300+z,12+z,z] = 3.125
            full[0,0,0] = -0.
            full_img = nib.Nifti1Image(full, affine)
            full_by_id[cid] = full
        full_img.header.set_xyzt_units('mm')
        original = full_img
        if not metadata_only and i in (0,1):
            target = ('S','R','A') if i == 0 else ('R','A','S')
            orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),
                                                         nib.orientations.axcodes2ornt(target))
            original = full_img.as_reoriented(orientation)
        folder = root/'cases'/cid; folder.mkdir(parents=True)
        sha = hashlib.sha256(cid.encode()).hexdigest()
        record = dict(schema=t.SCHEMA, status='COMPLETE', case_id=cid, depth=depth,
            partition_fold0=partitions[cid], source_ct_sha256=sha, source_gt_sha256=sha,
            full_lps_geometry=geom(full_img), original_ct_geometry=geom(original),
            original_gt_geometry=geom(original), tiles=[])
        if metadata_only:
            for key in ('full_lps_geometry','original_ct_geometry','original_gt_geometry'):
                record[key]['shape_xyz']=[512,512,depth]
        for tid in range(16):
            row,col = divmod(tid,4); crop=dict(x0=col*128,x1=(col+1)*128,y0=row*128,y1=(row+1)*128)
            shift = np.eye(4);shift[:3,3]=[crop['x0'],crop['y0'],0]; a=affine@shift
            tile=dict(tile_id=tid,row=row,col=col,xyz_crop=crop,shape_xyz=[128,128,depth],affine=a.tolist())
            for role,suffix,dtype in [('ct','ct','float32'),('gt','seg','uint16')]:
                name=f'{cid}_t{tid:02d}_{suffix}.nii.gz';path=folder/name
                if metadata_only:
                    path.touch()
                else:
                    values=full[crop['x0']:crop['x1'],crop['y0']:crop['y1']]
                    if role=='gt': values=(values>0).astype(np.uint16)
                    img=nib.Nifti1Image(values.astype(dtype),a);img.header.set_xyzt_units('mm')
                    img.set_sform(a,code=2);img.set_qform(None,code=0);nib.save(img,path)
                tile[role]=dict(file=name,sha256=t.file_sha256(path),dtype=dtype,bytes=path.stat().st_size)
            record['tiles'].append(tile)
        dump(folder/'case_export.json',record)
        rows.append(dict(case_id=cid,depth=depth,partition_fold0=partitions[cid],case_manifest=f'cases/{cid}/case_export.json'))
        source_files.append(dict(case_id=cid, partition=partitions[cid],ct_output_sha256=sha,gt_output_sha256=sha))
    train_count=len(split['folds'][0]['train']['case_ids'])
    dump(meta/'EXPORT_COMPLETE.json',dict(status='COMPLETE_WITH_PROVENANCE_LIMITATION',
        policy='train_fovaware_clipped_zscore_v1',count=n,train_count=train_count,files=source_files,
        note='SYNTHETIC SOFTWARE FIXTURE; not clinical data'))
    manifest=dict(schema=t.SCHEMA,status='COMPLETE',source='ALREADY_NORMALIZED_V3_FOLD0',
        normalization_applied_during_tiling=False,cases_exported=n,ct_tile_files=16*n,gt_tile_files=16*n,
        master_slices=sum(depths),tile_masters=16*sum(depths),cases=rows)
    dump(root/'TILE_MANIFEST.json',manifest)
    dump(root/'READY_ALL_199.json',dict(status='PASS',cases=n,tile_files=n*32,fold_scope='FOLD0_ONLY',
        tile_manifest_sha256=t.file_sha256(root/'TILE_MANIFEST.json')))
    counts=(n,sum(depths),tuple(len(split['folds'][0][p]['case_ids']) for p in ('train','validation','test')))
    return root,full_by_id,counts


@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    return fixture_cohort(tmp_path_factory.mktemp('b3b')/'tiles')


@pytest.fixture
def bundle(cohort, monkeypatch):
    root,full,counts=cohort
    monkeypatch.setattr(t,'COHORT_COUNTS',counts)
    return t.TileBundle(root,assume_mm=True)


def experiment(root, tmp_path, *, compact=True):
    path=ROOT/'configs/merit_tiling128_v3_fold0.yaml'
    config=load_experiment(path,tile_root=root)
    if compact:
        # Only the synthetic software fixture is compact. Production YAML stays 16/[3,3,3,3].
        config['model'].update(feature_size=2,depths=[1,1,1,1])
        config['training'].update(device='cpu',amp=False,epochs=1,include_hd95=False,threads=2)
    return validate_experiment(config)


def test_complete199_counts_metadata_only(tmp_path):
    root,_,counts=fixture_cohort(tmp_path/'metadata199',199,metadata_only=True)
    assert counts==(199,13705,(127,32,40))
    b=t.TileBundle(root,assume_mm=True)
    assert len(b.cases)==199 and len(t.TiledCTSlabDataset(b.cases,PreprocessConfig(assume_mm=True)))==219280
    assert b.manifest['ct_tile_files']==b.manifest['gt_tile_files']==3184


def test_source_fingerprint_preserved_without_original_niftis(bundle):
    assert not bundle.source_cases[0].image.exists()
    assert bundle.split['manifest_fingerprint']==json.loads(bundle.split_path.read_text())['manifest_fingerprint']
    assert len(bundle.cases)==10


def test_shared_tiled_loader_entry_point(bundle):
    from merit_covid.data.pipeline.dataloader import build_tiled_fold_loaders
    loaders=build_tiled_fold_loaders(bundle.root,preprocess=PreprocessConfig(assume_mm=True))
    assert [len(getattr(loaders,p).dataset.cases) for p in ('train','validation','test')]==[6,2,2]
    assert isinstance(loaders.train.sampler,VolumeShuffleSampler)
    assert loaders.validation.shuffle_mode==loaders.test.shuffle_mode=='sequential'


@pytest.mark.parametrize('index',[0,1,2])
def test_exact_all_slabs_restore_gt_geometry_and_boundaries(bundle,cohort,index):
    case=bundle.cases[index];full=cohort[1][case.case_id];depth=case.record['depth']
    ds=t.TiledCTSlabDataset([case],PreprocessConfig(assume_mm=True))
    assert len(ds)==16*depth and ds.ends==[0,16*depth]
    for i in range(len(ds)):
        sample=ds[i];tid,z=divmod(i,depth);row,col=divmod(tid,4)
        selected=np.clip(np.arange(z-5,z+6),0,depth-1)
        expected=full[col*128:(col+1)*128,row*128:(row+1)*128,:].transpose(2,1,0)[selected]
        np.testing.assert_array_equal(sample['image_slab'][0].numpy().view('uint32'),expected.view('uint32'))
        np.testing.assert_array_equal(sample['mask_slab'][0],(expected>0).astype(np.float32))
        assert sample['image_slab'].shape==(1,11,128,128) and sample['master_index']==5
        assert int(sample['attention_valid'].sum())==len(set(selected.tolist()))
        assert int(sample['slice_valid'].sum())==len(set(selected.tolist()))
        assert sample['metadata']['full_volume_metadata']['canonical_shape_xyz']==[512,512,depth]
        np.testing.assert_allclose(sample['slice_positions_mm'][5],
            nib.affines.apply_affine(case.record['full_lps_geometry']['affine'],[col*128+63.5,row*128+63.5,z]),atol=1e-4)
    gt,orig=t.reconstruct_gt(case,assume_mm=True)
    np.testing.assert_array_equal(gt,full.transpose(2,1,0)>0)
    pred=VolumePrediction(case.case_id,gt,case.metadata)
    np.testing.assert_array_equal(pred.to_original_xyz(),orig)
    expected_original=nib.orientations.apply_orientation((full>0).astype(np.float32),np.array(case.metadata['canonical_to_original_orientation']))
    np.testing.assert_array_equal(orig,expected_original)


def test_local_gt_is_not_master_repeated(bundle):
    ds=t.TiledCTSlabDataset([bundle.cases[1]],PreprocessConfig(assume_mm=True))
    sample=ds[2*3+1]  # col2/row0: tile with per-Z moving lesion
    assert not torch.equal(sample['mask_slab'][0,4],sample['mask_slab'][0,5])


def test_dataloader_sampler_cache_and_resume(bundle):
    ds=t.TiledCTSlabDataset(bundle.cases,PreprocessConfig(assume_mm=True));ds.partition='train'
    sample=ds[0];state=pickle.loads(pickle.dumps(ds));assert state._cached is None
    loader=make_dataloader(ds,batch_size=2,shuffle=True,shuffle_mode='volume',seed=42)
    rng=loader.generator.get_state().clone()
    order=list(loader.sampler);loader.generator.set_state(rng)
    assert order==list(loader.sampler) and sorted(order)==list(range(len(ds)))
    case_indices=[ds.sample_key(i)[0] for i in order]
    assert sum(a!=b for a,b in zip(case_indices,case_indices[1:]))==len(ds.cases)-1
    bal=VolumeBalancedSampler(ds,3,torch.Generator().manual_seed(3));assert len(list(bal))==3*len(ds.cases)
    batch=collate_slabs([ds[0],ds[1]])
    assert batch['image_slab'].shape==(2,1,11,128,128)
    assert batch['mask_slab'].dtype==torch.float32
    # Replay sampler positions without forcing a large image iteration.
    generator=torch.Generator().manual_seed(9)
    tiny=torch.utils.data.DataLoader(list(range(40)),batch_size=2,shuffle=True,generator=generator)
    before=generator.get_state().clone();it=iter(tiny);next(it);next(it)
    checkpoint=generator.get_state().clone();expected=torch.cat(list(it));generator.set_state(before)
    assert torch.equal(torch.cat(list(resume_loader_iterator(tiny,2,checkpoint,lambda:None))),expected)


def test_probability_assembly_exact_and_negative_cases(bundle,cohort):
    case=bundle.cases[1];depth=case.record['depth']
    expected=np.random.default_rng(4).random((depth,512,512),dtype=np.float32)
    assembly=TileProbabilityReconstructor(case)
    for tile in reversed(case.record['tiles']):
        crop=tile['xyz_crop']
        for z in reversed(range(depth)):
            assembly.add(case.case_id,tile['tile_id'],z,crop,expected[z,crop['y0']:crop['y1'],crop['x0']:crop['x1']])
    np.testing.assert_array_equal(assembly.finish(),expected)
    with pytest.raises(ValueError,match='duplicate'):assembly.add(case.case_id,0,0,case.record['tiles'][0]['xyz_crop'],np.zeros((128,128)))
    for cid,tid,z,crop,p in [('wrong',0,0,{},np.zeros((128,128))),
                          (case.case_id,16,0,{},np.zeros((128,128))),
                          (case.case_id,0,depth,{},np.zeros((128,128))),
                          (case.case_id,0,0,dict(x0=1),np.zeros((128,128))),
                          (case.case_id,0,0,case.record['tiles'][0]['xyz_crop'],np.zeros((127,128))),
                          (case.case_id,0,0,case.record['tiles'][0]['xyz_crop'],np.full((128,128),np.nan))]:
        with pytest.raises(ValueError):TileProbabilityReconstructor(case).add(cid,tid,z,crop,p)
    with pytest.raises(ValueError,match='Missing'):TileProbabilityReconstructor(case).finish()


def fake_ct_model():
    model=MERITUNETRPP(img_size=(11,128,128),feature_size=2,depths=(1,1,1,1))
    def forward(self,x,master,z_offsets_mm=None,attention_valid=None):
        logits=torch.cat([torch.zeros_like(x),x-.01],1)
        return MERITOutput(logits,None,None,logits)
    model.forward=types.MethodType(forward,model)
    return model


def test_ct_only_inference_never_opens_gt_and_restores_full_grid(bundle,cohort,monkeypatch):
    from merit_covid.engine.slab_inference import predict_slab_case
    case=bundle.cases[0];actual_load=nib.load;opened=[]
    def guarded(path,*a,**kw):
        assert '_seg' not in str(path),'GT opened during inference'
        opened.append(str(path));return actual_load(path,*a,**kw)
    model=fake_ct_model().eval();torch.set_num_threads(2)
    with monkeypatch.context() as context:
        context.setattr(nib,'load',guarded)
        prediction,presence=predict_slab_case(model,case,PreprocessConfig(assume_mm=True),batch_size=4,return_presence=True)
    assert len(set(opened))==16 and presence is None
    assert prediction.probability_zyx.shape==(1,512,512)
    expected=(torch.tensor(cohort[1][case.case_id].transpose(2,1,0))-.01).sigmoid().numpy()
    np.testing.assert_array_equal(prediction.probability_zyx,expected)
    assert prediction.to_original_xyz().shape==tuple(case.metadata['original_shape_xyz'])


def test_full_volume_evaluation_dice_iou_hd95_no_tile_rows(bundle):
    model=fake_ct_model();torch.set_num_threads(2)
    report=evaluate_model(model,[bundle.cases[0]],PreprocessConfig(assume_mm=True),include_hd95=True,batch_size=4)
    assert model.training and len(report['per_case'])==1
    row=report['per_case'][0]
    assert row['case_id']==bundle.cases[0].case_id and row['dice']==row['iou']==1.
    assert row['hd95_mm']==0. and row['lesion_slice_detail']['presence'] is None
    assert report['whole_slice_presence_status'].startswith('UNSUPPORTED')


@pytest.mark.parametrize('mutation',['missing','duplicate','tileid','caseid','crop','depth','affine','traversal','source_sha','original_geometry'])
def test_bad_case_manifests_rejected(bundle,tmp_path,mutation,monkeypatch):
    # Redirect JSON reads only; preserve shared fixture bytes.
    original=t.read_json;target=bundle.cases[0].manifest
    record=deepcopy(original(target));tiles=record['tiles']
    if mutation=='missing':tiles.pop()
    if mutation=='duplicate':tiles[-1]=deepcopy(tiles[0])
    if mutation=='tileid':tiles[0]['tile_id']=1
    if mutation=='caseid':record['case_id']='WRONG'
    if mutation=='crop':tiles[0]['xyz_crop']['x0']=1
    if mutation=='depth':record['depth']+=1
    if mutation=='affine':tiles[0]['affine'][0][3]+=1
    if mutation=='traversal':tiles[0]['ct']['file']='../../escape.nii.gz'
    if mutation=='source_sha':record['source_ct_sha256']='0'*64
    if mutation=='original_geometry':record['original_ct_geometry']['affine'][0][3]+=1
    monkeypatch.setattr(t,'read_json',lambda p:record if Path(p)==target else original(p))
    with pytest.raises(ValueError):t.TileBundle(bundle.root,assume_mm=True)


def test_bad_ready_raw_and_source_fingerprint_rejected(bundle,monkeypatch):
    original=t.read_json
    for path,mutate in [('READY_ALL_199.json',lambda r:r.update(tile_manifest_sha256='0'*64)),
                        ('TILE_MANIFEST.json',lambda r:r.update(source='RAW')),
                        ('source_metadata/splits_covid19_20_5fold.json',lambda r:r.update(manifest_fingerprint='WRONG'))]:
        target=bundle.root/path;record=deepcopy(original(target));mutate(record)
        with monkeypatch.context() as context:
            context.setattr(t,'read_json',lambda p:record if Path(p)==target else original(p))
            with pytest.raises(ValueError):t.TileBundle(bundle.root,assume_mm=True)


@pytest.mark.parametrize('fold',[1,2,3,4,-1,True])
def test_fold_leakage_fail_fast(bundle,tmp_path,fold):
    config=experiment(bundle.root,tmp_path)
    with pytest.raises(ValueError,match='FOLD0_ONLY'):split_datasets(config,fold)


@pytest.mark.parametrize('change',[{'intensity_encoding':'train_zscore','zscore_mean':0,'zscore_std':1},
                                  {'size_hw':[128,128]},{'boundary_policy':'reflect'},
                                  {'duplicate_slice_policy':'all'},{'slab_size':5}])
def test_wrong_preprocessing_config_rejected(bundle,tmp_path,change):
    config=experiment(bundle.root,tmp_path);config['preprocess'].update(change)
    with pytest.raises(ValueError):validate_experiment(config)


def test_missing_mask_ct_only_and_preflight_binary_hash_headers(bundle,monkeypatch):
    case=bundle.cases[0];target=case.path(0,'gt');actual=Path.is_file
    with monkeypatch.context() as context:
        context.setattr(Path,'is_file',lambda p:False if p==target else actual(p))
        with pytest.raises(FileNotFoundError):t.TileBundle(bundle.root,assume_mm=True)
        assert t.TileBundle(bundle.root,assume_mm=True,require_mask=False)
    report=bundle.preflight(case_ids=[case.case_id],hash_files=True,voxels=True)
    assert report['files_checked']==32 and report['status']=='PASS'
    actual_sha=t.file_sha256
    with monkeypatch.context() as context:
        context.setattr(t,'file_sha256',lambda p:'0'*64 if Path(p)==target else actual_sha(p))
        with pytest.raises(ValueError,match='SHA'):bundle.preflight(case_ids=[case.case_id],hash_files=True)
    actual_load=nib.load
    def corrupt(path,*args,**kwargs):
        image=actual_load(path,*args,**kwargs)
        if Path(path)==target:
            class CorruptGT:
                def get_fdata(self,**kw):return np.full(image.shape,2.,np.float64)
            return CorruptGT()
        return image
    # Inject decoded labels, after header validation, rather than rewriting data.
    with monkeypatch.context() as context:
        context.setattr(t,'inspect_tile',lambda c,i,role,mm:corrupt(c.path(i,role)))
        with pytest.raises(ValueError,match='binary'):t.reconstruct_gt(case,assume_mm=True)


def test_train_config_split_actual_backward_checkpoint_resume_and_source_binding(bundle,tmp_path):
    from merit_covid.engine.tiled_smoke import tile_forward_backward
    config=experiment(bundle.root,tmp_path)
    datasets,provenance=split_datasets(config,0)
    assert len(provenance['all_image_paths'])==len(provenance['all_mask_paths'])==160
    assert [len(datasets[p].cases) for p in ('train','validation','test')]==[6,2,2]
    assert len({c.case_id for d in datasets.values() for c in d.cases})==10
    smoke=tile_forward_backward(config,bundle,bundle.cases[0].case_id,'cpu')
    assert smoke['finite_gradients'] and smoke['optimizer_updates']==0 and smoke['input_shape']==[1,1,11,128,128]
    loader=make_dataloader(datasets['train'],shuffle=True,shuffle_mode='volume',seed=config['training']['seed'])
    trainer=Trainer(config,loader,datasets['validation'].cases,provenance,tmp_path/'run')
    # One bounded optimizer update exercises the actual train loop; no full epoch.
    assert trainer.train_epoch(0,deadline=time.monotonic()-1) is None
    checkpoint=tmp_path/'run/in_progress.pt';assert checkpoint.is_file()
    expected={k:v.clone() for k,v in trainer.model.state_dict().items()}
    restored=Trainer(config,make_dataloader(datasets['train'],shuffle=True,shuffle_mode='volume',seed=config['training']['seed']),
                     datasets['validation'].cases,provenance,tmp_path/'resume')
    restored.resume(checkpoint)
    assert restored._partial_progress['batches']==1
    assert all(torch.equal(expected[k],v) for k,v in restored.model.state_dict().items())
    saved=torch.load(checkpoint,weights_only=False)
    actual=restored.optimizer.state_dict()
    assert actual['param_groups']==saved['optimizer']['param_groups']
    for key,values in saved['optimizer']['state'].items():
        for name,value in values.items():
            assert torch.equal(actual['state'][key][name],value) if isinstance(value,torch.Tensor) else actual['state'][key][name]==value
    assert restored.scheduler.state_dict()==saved['scheduler']
    assert restored.scaler.state_dict()==saved['scaler']
    assert torch.equal(torch.get_rng_state(),saved['rng']['torch'])
    assert torch.equal(restored.train_loader.generator.get_state(),saved['loader_generator'])
    assert restored.train_epoch(0,deadline=time.monotonic()-1) is None
    assert restored._partial_progress is None
    cp=torch.load(checkpoint,weights_only=False)
    # Synthetic checkpoint metadata for the other protocol; rejected BEFORE any
    # shape-dependent weights can be partially loaded.
    cp['config']['data'].pop('mode');cp['config']['data'].pop('tile_root')
    cp['config']['model']['img_size']=[11,512,512]
    cp['protocol_fingerprint']=protocol_fingerprint(cp['config'],cp['provenance'])
    with pytest.raises(ValueError,match='Native512'):restored.resume(cp)
    bad_config=deepcopy(config);bad_config['data']['manifest']=str(tmp_path/'wrong.json')
    with pytest.raises(ValueError,match='original metadata'):validate_experiment(bad_config)
    changed=deepcopy(provenance);changed['source_identity']['tile_manifest_sha256']='0'*64
    assert protocol_fingerprint(config,changed)!=trainer.fingerprint
    model,payload=load_checkpoint_model(checkpoint)
    assert model.img_size==(11,128,128) and payload['provenance']['fold']==0


def test_native512_config_and_model_source_unchanged(bundle,tmp_path):
    native=load_experiment(ROOT/'configs/merit_native512.yaml')
    assert native['model']['img_size']==[11,512,512] and native['preprocess']['size_hw'] is None
    tiled=load_experiment(ROOT/'configs/merit_tiling128_v3_fold0.yaml',tile_root=bundle.root)
    for key in ('feature_size','depths','use_tep','use_mer','use_pcg'):
        assert native['model'][key]==tiled['model'][key]
    assert native['loss']==tiled['loss'] and native['training']==tiled['training']


@pytest.mark.parametrize('workers',[0,2])
def test_worker_collation_keeps_tile_geometry(bundle,workers):
    ds=t.TiledCTSlabDataset([bundle.cases[0]],PreprocessConfig(assume_mm=True))
    loader=make_dataloader(ds,batch_size=2,num_workers=workers)
    iterator=iter(loader);batch=next(iterator)
    assert batch['image_slab'].shape==(2,1,11,128,128)
    assert batch['tile_id'].tolist()==[0,1]
    assert batch['metadata'][1]['xyz_crop']['x0']==128
    del iterator


@pytest.mark.parametrize('kind',['shape','affine','dtype','nonfinite_ct'])
def test_actual_tile_header_and_voxel_failures(bundle,monkeypatch,kind):
    case=bundle.cases[0];path=case.path(0,'ct');actual=nib.load
    image=actual(path)
    values=image.get_fdata(dtype=np.float32)
    a=image.affine.copy()
    if kind=='shape':values=values[:-1]
    if kind=='affine':a[0,3]+=5
    if kind=='dtype':values=values.astype(np.float64)
    if kind=='nonfinite_ct':values[0,0,0]=np.nan
    bad=nib.Nifti1Image(values,a);bad.header.set_xyzt_units('mm')
    # Array-backed NIfTIs lack ArrayProxy slope/inter fields; load/save in scratch
    # is tested separately by preflight; this hook supplies a real NIfTI proxy.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='b3b_bad_header_') as folder:
        bad_path=Path(folder)/'bad.nii.gz';nib.save(bad,bad_path)
        monkeypatch.setattr(nib,'load',lambda p,*args,**kw:actual(bad_path) if Path(p)==path else actual(p,*args,**kw))
        ds=t.TiledCTSlabDataset([case],PreprocessConfig(assume_mm=True))
        with pytest.raises(ValueError):ds[0]


def test_cli_preflight_check_split_cv_guard_train_and_evaluate_routes(bundle,tmp_path,monkeypatch):
    import sys
    from scripts import merit_pipeline as cli
    common=['--config',str(ROOT/'configs/merit_tiling128_v3_fold0.yaml'),'--tile-root',str(bundle.root)]
    output=tmp_path/'preflight.json'
    monkeypatch.setattr(sys,'argv',['merit_pipeline','preflight',*common,'--case-id',bundle.cases[0].case_id,
                                    '--hash-files','--voxels','--output',str(output)])
    cli.main();report=json.loads(output.read_text())
    assert report['status']=='PASS' and not report['full_cohort_qc']
    monkeypatch.setattr(sys,'argv',['merit_pipeline','cv',*common,'--all-folds','--output',str(tmp_path/'cv')])
    with pytest.raises(ValueError,match='FOLD0_ONLY'):cli.main()
    called=[]
    monkeypatch.setattr(cli,'train_fold',lambda config,fold,output,resume:called.append((config,fold)))
    monkeypatch.setattr(sys,'argv',['merit_pipeline','train',*common,'--fold','0','--output',str(tmp_path/'not_trained')])
    cli.main();assert called and called[0][1]==0
    # Only the command route is mocked; the full-volume evaluator itself ran above.
    config=experiment(bundle.root,tmp_path);datasets,prov=split_datasets(config,0)
    cp=dict(config=config,provenance=prov,protocol_fingerprint='synthetic-cli-route')
    path=tmp_path/'cli_checkpoint.pt';path.write_bytes(b'CLI route fixture only')
    monkeypatch.setattr(cli,'load_checkpoint_model',lambda path,device:(fake_ct_model().eval(),cp))
    monkeypatch.setattr(cli,'evaluate_model',lambda model,cases,preprocess,**kw:
        {'per_case':[{'case_id':c.case_id} for c in cases],'synthetic_route_only':True})
    output=tmp_path/'evaluate.json'
    monkeypatch.setattr(sys,'argv',['merit_pipeline','evaluate','--checkpoint',str(path),'--partition','validation','--output',str(output)])
    cli.main();assert len(json.loads(output.read_text())['per_case'])==2


def test_real_cli_bounded_synthetic_train_and_full_volume_evaluate(bundle,tmp_path):
    """Execute the real commands, not a mocked CLI route, with ONE update."""
    import subprocess,sys
    config=experiment(bundle.root,tmp_path)
    path=tmp_path/'synthetic_cli.yaml';path.write_text(yaml.safe_dump(config))
    run=tmp_path/'new_run'
    # The child gets the SAME in-memory synthetic count override as pytest's
    # fixture, not a production config/environment bypass. CLI main/train/eval
    # are real. Strict 199 counts have their own unmodified-parser test.
    counts=(len(bundle.cases),bundle.manifest['master_slices'],(6,2,2))
    driver=("import sys,runpy;sys.path.insert(0,'src');sys.path.insert(0,'.');"
            "from merit_covid.data.pipeline import tiled;"
            f"tiled.COHORT_COUNTS={counts!r};"
            "sys.argv=['scripts/merit_pipeline.py']+sys.argv[1:];"
            "runpy.run_path('scripts/merit_pipeline.py',run_name='__main__')")
    subprocess.run([sys.executable,'-c',driver,'train','--config',str(path),
                    '--fold','0','--output',str(run),'--checkpoint-every','1','--max-seconds','0.00000001'],
                   cwd=ROOT,check=True,capture_output=True,text=True,timeout=90)
    checkpoint=run/'in_progress.pt';cp=torch.load(checkpoint,weights_only=False)
    assert cp['in_progress']['batches']==cp['in_progress']['steps']==1
    assert not (run/'last.pt').exists()  # No complete synthetic epoch.
    report=tmp_path/'volume_validation.json'
    subprocess.run([sys.executable,'-c',driver,'evaluate','--checkpoint',str(checkpoint),
                    '--partition','validation','--device','cpu','--output',str(report)],
                   cwd=ROOT,check=True,capture_output=True,text=True,timeout=90)
    result=json.loads(report.read_text())
    assert {r['case_id'] for r in result['per_case']}==set(cp['provenance']['partitions']['validation']['case_ids'])
    assert len(result['per_case'])==2 and 'full original' in result['evaluation_grid']
    assert all(np.isfinite(row['dice']) and np.isfinite(row['iou']) for row in result['per_case'])
