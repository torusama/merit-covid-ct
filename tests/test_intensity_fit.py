"""CPU preprocessing only: no model, optimizer, CUDA calls or metric evaluation."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.intensity_fit import (POLICY, RULE, QuantizedHistogram, digest,
    identify_fit_padding, accepted_blocks, clipped_moments, fit_population,
    validate_fit_stats, validate_fit_binding)
from merit_covid.data.pipeline.preprocessing import read_arrays, normalize_ct
from merit_covid.data.pipeline.geometry import restore_volume
from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.data.pipeline.dataloader import make_dataloader


def example_stats():
    ids = [f'TRAIN-{i:03d}' for i in range(127)]
    stats = dict(schema_version=1, status='FITTED', policy=POLICY,
        parameters=dict(lower=-1000.,upper=500.,mean=-400.,std=250.),fold=0,
        case_ids=ids,case_ids_sha256=digest(ids),split_sha256_lf='a'*64,
        manifest_sha256='b'*64,manifest_fingerprint='c'*64,padding_rule=RULE,
        voxel_counts=dict(total=2000,accepted=1800,excluded=200,sentinel=201),
        source_ct_sha256={i:'d'*64 for i in ids})
    stats['stats_sha256']=digest(stats)
    return stats


def policy(stats=None):
    stats = stats or example_stats()
    return PreprocessConfig(intensity_encoding=POLICY,zscore_fit=stats,
        zscore_mean=stats['parameters']['mean'],zscore_std=stats['parameters']['std'],assume_mm=True)


def padded(n=64, depth=4):
    ct=np.ones((n,n,depth),dtype=np.float32)*50
    ct[:4,:,:]=ct[-4:,:,:]=ct[:,:4,:]=ct[:,-4:,:]=-2048
    return ct


def test_boundary_padding_keeps_isolated_interior_and_non_circle():
    ct=padded();ct[20,20,1]=-2048
    before=ct.copy();mask,row=identify_fit_padding(ct)
    assert row['status']=='CONFIDENT' and row['excluded_voxels']==int(mask.sum())*4
    assert not mask[20,20] and row['isolated_sentinel_voxels_kept']==1
    assert row['boundary_components']==1  # A rectangular band, no circle fit.
    np.testing.assert_array_equal(ct,before)


@pytest.mark.parametrize('kind', ['none','minus1024','deep','varying','extra_component','short'])
def test_unknown_or_ambiguous_falls_back_to_all_voxels(kind):
    ct=padded()
    if kind=='none':ct[:]=50
    if kind=='minus1024':ct[ct==-2048]=-1024
    if kind=='deep':ct[:33,:,:]=-2048
    if kind=='varying':ct[:5,:,1]=-2048
    if kind=='extra_component':
        ct=padded();ct[20:40,0:15,:]=-2048
    if kind=='short':ct=ct[:,:,:2]
    mask,row=identify_fit_padding(ct)
    assert mask is None and row['status'] in ('UNKNOWN','AMBIGUOUS')
    assert row['excluded_voxels']==0 and row['unverified_background_included'] is True
    assert sum(v.size for v in accepted_blocks(ct,mask))==ct.size


def test_quantiles_and_moments_use_only_the_accepted_population():
    ct=padded();ct[4:-4,4:-4,:]=np.arange(56*56*4,dtype=np.float32).reshape(56,56,4)/10
    ct[20,20,1]=-2048  # Isolated interior remains part of the fit population.
    mask,_=identify_fit_padding(ct)
    blocks=list(accepted_blocks(ct,mask,2))
    hist=QuantizedHistogram()
    for b in blocks:hist.add(b)
    population=np.concatenate(blocks)  # Tiny test oracle only, never production fit.
    lower,upper=hist.percentile(.1),hist.percentile(99.9)
    oracle=np.percentile(population.astype(np.float64),[.1,99.9])
    np.testing.assert_allclose([lower,upper],oracle,atol=1/32)
    n,mu,m2=clipped_moments(iter(blocks),lower,upper)
    expected=np.clip(population.astype(np.float64),lower,upper)
    assert n==len(expected)
    assert mu==pytest.approx(expected.mean(),abs=1e-10)
    assert np.sqrt(m2/n)==pytest.approx(expected.std(),abs=1e-10)


def test_two_pass_fit_and_widespread_fallback_gate():
    cases=[SimpleNamespace(case_id=str(i)) for i in range(5)]
    arrays={c.case_id:padded()+i for i,c in enumerate(cases)}
    # Preserve exact sentinel while varying only accepted CT values.
    for i,c in enumerate(cases):
        arrays[c.case_id][padded()==-2048]=-2048
    params,rows,blocked=fit_population(cases,lambda c:arrays[c.case_id])
    assert not blocked and params['lower']<params['upper'] and params['std']>0
    assert len(rows)==5 and all(r['status']=='CONFIDENT' for r in rows)
    params,rows,blocked=fit_population(cases,lambda c:np.ones((16,16,4),dtype=np.float32))
    assert params is None and blocked and all(r['status']=='UNKNOWN' for r in rows)


def test_transform_all_voxels_is_deterministic_float32_and_no_detection(monkeypatch):
    import merit_covid.data.pipeline.intensity_fit as module
    monkeypatch.setattr(module,'identify_fit_padding',lambda *_:pytest.fail('No padding detector at inference'))
    x=np.array([-9653.,-2048.,-1024.,-400.,0.,500.,32737.],dtype=np.float32)
    before=x.copy();cfg=policy()
    expected=((np.clip(x.astype(np.float64),-1000,500)+400)/250).astype(np.float32)
    y=normalize_ct(x,cfg,'CT-only')
    np.testing.assert_array_equal(y,expected)
    np.testing.assert_array_equal(y,normalize_ct(x,cfg,'another-CT'))
    np.testing.assert_array_equal(x,before)
    assert y.dtype==np.float32 and np.isfinite(y).all() and y[1]!=0


@pytest.mark.parametrize('change', ['status','checksum','ids','std','bounds','rule'])
def test_unfitted_tampered_or_invalid_stats_rejected(change):
    stats=example_stats()
    if change=='status':stats['status']='NOT_FITTED'
    if change=='checksum':stats['parameters']['mean']=20
    if change=='ids':stats['case_ids'].append('VAL-001')
    if change=='std':stats['parameters']['std']=0.
    if change=='bounds':stats['parameters']['lower']=1000.
    if change=='rule':stats['padding_rule']={}
    if change!='checksum':stats['stats_sha256']=digest({k:v for k,v in stats.items() if k!='stats_sha256'})
    with pytest.raises(ValueError):validate_fit_stats(stats)


def test_actual_manifest_split_and_fit_selection_binding(tmp_path):
    stats=example_stats();manifest=tmp_path/'cases.json';manifest.write_text('[]')
    split=tmp_path/'split.json'
    data=dict(manifest_fingerprint=stats['manifest_fingerprint'],folds=[dict(fold=0,train=dict(case_ids=stats['case_ids']))])
    split.write_text(json.dumps(data))
    stats['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    stats['split_sha256_lf']=hashlib.sha256(split.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    stats['stats_sha256']=digest({k:v for k,v in stats.items() if k!='stats_sha256'})
    train=[SimpleNamespace(case_id=i) for i in stats['case_ids']]
    validate_fit_binding(stats,manifest,split,cases=train)
    with pytest.raises(ValueError,match='TRAIN'):
        validate_fit_binding(stats,manifest,split,cases=[*train[:-1],SimpleNamespace(case_id='VAL-001')])
    with pytest.raises((ValueError,StopIteration)):
        validate_fit_binding(stats,manifest,split,fold=1)
    manifest.write_text('[{}]')
    with pytest.raises(ValueError):validate_fit_binding(stats,manifest,split)


@pytest.mark.parametrize('lpi', [False,True])
def test_gt_alignment_coordinates_loader_and_ct_only_inference(tmp_path,lpi):
    ct=np.arange(512*512*3,dtype=np.float32).reshape(512,512,3)%1500-1100
    mask=np.zeros_like(ct);mask[7,25,0]=mask[301,13,2]=1
    affine=np.diag([-0.7,-0.7,-5. if lpi else 5.,1.])
    a=tmp_path/'ct.nii.gz';b=tmp_path/'mask.nii.gz'
    for array,path in ((ct,a),(mask,b)):
        image=nib.Nifti1Image(array,affine);image.header.set_xyzt_units('mm');nib.save(image,path)
    before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (a,b)]
    case=Case('example',a,b,'example')
    native,native_gt,metadata=read_arrays(case,PreprocessConfig(assume_mm=True))
    processed,gt,meta=read_arrays(case,policy())
    np.testing.assert_array_equal(gt,native_gt)
    np.testing.assert_array_equal(restore_volume(gt,meta),mask)
    np.testing.assert_array_equal(np.argwhere(gt==1),np.argwhere(native_gt==1))
    assert meta['canonical_orientation']==['L','P','S'] and int(gt.sum())==2
    ds=CTSlabDataset([case],policy());batch=next(iter(make_dataloader(ds,batch_size=1)))
    assert batch['image_slab'].shape==(1,1,11,512,512)
    np.testing.assert_array_equal(batch['master_mask'][0,0].numpy(),gt[0])
    # Missing file at inference must remain legal, with no GT reads.
    only=CTSlabDataset([Case('example',a,tmp_path/'not-present.nii','example')],policy(),require_mask=False)
    sample=only[0];assert 'mask_slab' not in sample
    assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in (a,b)]


def test_legacy_config_serialization_is_unchanged():
    assert set(asdict(PreprocessConfig()))=={'slab_size','size_hw','hu_window','intensity_encoding','assume_mm',
        'sampling_mode','target_spacing_mm','boundary_policy','duplicate_slice_policy','intensity_provenance',
        'zscore_mean','zscore_std','zscore_fit'}
    native=np.array([-2048,100],dtype=np.float32)
    np.testing.assert_array_equal(normalize_ct(native,PreprocessConfig(),'legacy'),native)


@pytest.mark.parametrize('invalid', ['nonbinary_float64','nonfinite','misaligned'])
def test_invalid_gt_is_rejected_before_float32_rounding(tmp_path,invalid):
    ct=np.zeros((16,16,3),dtype=np.float32);mask=np.zeros(ct.shape,dtype=np.float64)
    affine=np.diag([-1.,-1.,1.,1.]);mask_affine=affine.copy()
    if invalid=='nonbinary_float64':mask[1,1,1]=1+1e-10
    if invalid=='nonfinite':mask[1,1,1]=np.nan
    if invalid=='misaligned':mask_affine[0,3]=5
    a,b=tmp_path/'ct.nii',tmp_path/'gt.nii'
    for data,path,matrix in ((ct,a,affine),(mask,b,mask_affine)):
        image=nib.Nifti1Image(data,matrix);image.header.set_xyzt_units('mm');nib.save(image,path)
    with pytest.raises(ValueError):read_arrays(Case('invalid',a,b,'invalid'),PreprocessConfig())


def test_experiment_loads_one_stats_file_and_rejects_unfitted_or_changed_grid(tmp_path):
    import yaml
    from merit_covid.engine.training import load_experiment,validate_experiment
    stats=example_stats()
    manifest=tmp_path/'cases.json';manifest.write_text('[]')
    split=tmp_path/'split.json'
    split.write_text(json.dumps(dict(manifest_fingerprint=stats['manifest_fingerprint'],
        folds=[dict(fold=0,train=dict(case_ids=stats['case_ids']))])))
    stats['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    stats['split_sha256_lf']=hashlib.sha256(split.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    stats['stats_sha256']=digest({k:v for k,v in stats.items() if k!='stats_sha256'})
    statfile=tmp_path/'stats.json';statfile.write_text(json.dumps(stats))
    root=Path(__file__).resolve().parents[1]
    config=load_experiment(root/'configs/merit_native512.yaml')
    config['data']=dict(manifest=str(manifest),split_path=str(split))
    config['preprocess']=dict(intensity_encoding=POLICY,zscore_fit='stats.json',slab_size=11,assume_mm=True)
    path=tmp_path/'experiment.yaml';path.write_text(yaml.safe_dump(config))
    actual=load_experiment(path)
    assert actual['preprocess']['zscore_mean']==stats['parameters']['mean']
    assert actual['preprocess']['zscore_std']==stats['parameters']['std']
    config['preprocess'].update(zscore_mean=None,zscore_std=None)
    path.write_text(yaml.safe_dump(config))
    assert load_experiment(path)['preprocess']['zscore_mean']==stats['parameters']['mean']
    wrong=deepcopy(actual);wrong['model']['img_size']=[11,128,128]
    with pytest.raises(ValueError,match='11x512x512'):validate_experiment(wrong)
    statfile.write_text(json.dumps(dict(status='NOT_FITTED')))
    with pytest.raises(ValueError,match='NOT_FITTED'):load_experiment(path)
