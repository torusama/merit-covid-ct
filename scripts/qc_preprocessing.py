"""Read-only CPU QC through the production V3 reader, one CT/GT pair at a time.

No model, optimizer, normalization fit or segmentation metrics are executed.
Existing reports and source files are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

import nibabel as nib
import numpy as np

from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.geometry import restore_volume
from merit_covid.data.pipeline.preprocessing import read_arrays
from merit_covid.data.pipeline.intensity_fit import POLICY, file_hash
from merit_covid.engine.training import load_experiment, load_development_split

FIELDS = ['case_id','group_id','partition','status','reason','ct_shape_xyz','gt_shape_xyz',
          'ct_orientation','gt_orientation','canonical_orientation','native_spacing_mm',
          'canonical_shape_zyx','ct_finite_after_transform','gt_labels','gt_voxels_before',
          'gt_voxels_canonical','gt_voxels_restored','gt_coordinate_mismatches',
          'ct_sha256_before','ct_sha256_after','gt_sha256_before','gt_sha256_after','seconds']


def qc_case(case, preprocess, partition):
    """Exact binary/count/position checks; retain no volume after this function."""
    started = time.monotonic()
    row = dict(case_id=case.case_id,group_id=case.group_id,partition=partition,status='FAIL',reason='')
    try:
        if not case.image.is_file() or case.mask is None or not case.mask.is_file():
            row.update(status='BLOCKED_MISSING_DATA',reason='Missing CT or GT')
            return row
        if (case.image.name not in (case.case_id+'_ct.nii',case.case_id+'_ct.nii.gz')
                or case.mask.name not in (case.case_id+'_seg.nii',case.case_id+'_seg.nii.gz')):
            raise ValueError('Manifest case ID does not match the pinned cohort CT/seg filenames')
        row.update(ct_sha256_before=file_hash(case.image),gt_sha256_before=file_hash(case.mask))
        original = nib.load(case.mask)
        # Scaled source dtype, never cast to float32 before checking exact labels.
        truth = np.asanyarray(original.dataobj)
        if not np.isfinite(truth).all() or not np.isin(truth,[0,1]).all():
            raise ValueError('Original scaled GT is not finite exact binary 0/1 before cast')
        row.update(gt_shape_xyz=list(truth.shape),gt_orientation=list(nib.aff2axcodes(original.affine)),
                   gt_voxels_before=int(np.count_nonzero(truth)))
        image, mask, metadata = read_arrays(case,preprocess,require_mask=True)
        if (list(image.shape) != list(mask.shape) or list(image.shape[1:]) != [512,512]
                or metadata['native_size_hw'] != [512,512] or metadata['processed_size_hw'] != [512,512]
                or metadata['canonical_orientation'] != ['L','P','S'] or metadata['z_resampled']):
            raise ValueError('Unexpected CT/GT LPS/native grid or resampling')
        if metadata['lesion_retention']['resize_applied']:
            raise ValueError('XY resize is forbidden in final V3 QC')
        finite = bool(np.isfinite(image).all())
        if not finite:
            raise ValueError('Normalized CT contains nonfinite values')
        if not np.isin(mask,[0,1]).all():
            raise ValueError('Canonical GT is not exact binary')
        row.update(ct_shape_xyz=metadata['original_shape_xyz'],ct_orientation=metadata['original_orientation'],
            canonical_orientation=metadata['canonical_orientation'],native_spacing_mm=metadata['native_spacing_mm'],
            canonical_shape_zyx=list(image.shape),ct_finite_after_transform=finite,
            gt_labels=metadata['label_values'],gt_voxels_canonical=int(np.count_nonzero(mask)))
        del image  # Release the largest array before allocating restored GT.
        restored = restore_volume(mask,metadata)
        if restored.shape != truth.shape:
            raise ValueError('Restored GT shape differs from original GT')
        mismatches = 0
        for z in range(0,truth.shape[2],8):
            mismatches += int(np.count_nonzero(restored[:,:,z:z+8] != truth[:,:,z:z+8]))
        row.update(gt_voxels_restored=int(np.count_nonzero(restored)),gt_coordinate_mismatches=mismatches)
        if mismatches or not row['gt_voxels_before'] == row['gt_voxels_canonical'] == row['gt_voxels_restored']:
            raise ValueError('GT lesion count/coordinates changed through production V3/inverse mapping')
        row.update(ct_sha256_after=file_hash(case.image),gt_sha256_after=file_hash(case.mask))
        if row['ct_sha256_before'] != row['ct_sha256_after'] or row['gt_sha256_before'] != row['gt_sha256_after']:
            raise ValueError('Source bytes changed during read-only QC')
        row['status']='PASS'
    except Exception as error:
        row.update(status='FAIL',reason=type(error).__name__+': '+str(error))
    finally:
        row['seconds']=time.monotonic()-started
    return row


def run(config_path,output):
    started=time.monotonic();output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    config_path=Path(config_path).resolve()
    config=load_experiment(config_path)
    preprocess=PreprocessConfig(**config['preprocess'])
    if (preprocess.intensity_encoding != POLICY or preprocess.size_hw is not None or preprocess.slab_size != 11
            or preprocess.sampling_mode != 'index' or config['model']['position_mode'] != 'index'
            or config['loss']['weighting_mode'] != 'index'):
        raise ValueError('Final handoff QC must keep the locked V3/index/native512 policy')
    raw_config = __import__('yaml').safe_load(config_path.read_text(encoding='utf-8'))
    stats_path=(config_path.parent/raw_config['preprocess']['zscore_fit']).resolve()
    protected={str(p):file_hash(p) for p in (stats_path,Path(config['data']['manifest']),Path(config['data']['split_path']))}
    summary=dict(status='RUNNING',scope='CPU production preprocessing QC, not model evaluation or fitting',
        expected_cases=199,partition_counts_expected=dict(train=127,validation=32,test=40),
        stats_file_sha256=file_hash(stats_path),stats_provenance_sha256=preprocess.zscore_fit['stats_sha256'],
        stats_parameters=preprocess.zscore_fit['parameters'],normalization_refitted=False,
        gpu_used=False,training_updates=0,dice_iou_computed=False,rows=[])
    try:
        cases,split,identity,_=load_development_split(config)
    except FileNotFoundError as error:
        summary.update(status='BLOCKED_MISSING_DATA',reason=str(error),qc_completed=0)
        (output/'all199_ct_gt_qc_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        with (output/'all199_ct_gt_qc.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
            writer.writerow(dict(case_id='NOT_INDEXED',status='BLOCKED_MISSING_DATA',reason=str(error)))
        return summary
    fold=next(f for f in split['folds'] if f['fold']==0)
    partitions={case_id:role for role in ('train','validation','test') for case_id in fold[role]['case_ids']}
    if len(cases)!=199 or len(partitions)!=199:
        raise ValueError('Expected exactly199 unique cases in saved fold0')
    summary['identity']=identity
    with (output/'all199_ct_gt_qc.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for index,case in enumerate(cases,1):
            row=qc_case(case,preprocess,partitions[case.case_id])
            summary['rows'].append(row)
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
            f.flush()
            print('QC',index,199,case.case_id,row['partition'],row['status'],row['reason'],flush=True)
            (output/'qc_progress.json').write_text(json.dumps(dict(completed=index,total=199,
                passed=sum(r['status']=='PASS' for r in summary['rows']),elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
            gc.collect()
    rows=summary.pop('rows')
    counts={role:dict(total=sum(r['partition']==role for r in rows),
                       passed=sum(r['partition']==role and r['status']=='PASS' for r in rows))
            for role in ('train','validation','test')}
    unchanged=all(file_hash(path)==value for path,value in protected.items())
    summary.update(status='PASS' if len(rows)==199 and all(r['status']=='PASS' for r in rows) and unchanged else 'FAIL',
        qc_completed=len(rows),passed=sum(r['status']=='PASS' for r in rows),
        failed=sum(r['status']!='PASS' for r in rows),partitions=counts,
        failures=[r for r in rows if r['status']!='PASS'],protected_metadata_unchanged=unchanged,
        gt_coordinate_mismatches=sum(r.get('gt_coordinate_mismatches',0) for r in rows),
        source_files_hash_unchanged=sum(r.get('ct_sha256_before')==r.get('ct_sha256_after')
                                      and r.get('gt_sha256_before')==r.get('gt_sha256_after') for r in rows),
        elapsed_seconds=time.monotonic()-started,
        identity_limit='Manifest/filenames and geometry; patient independence/HU not established')
    (output/'all199_ct_gt_qc_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/merit_preprocessing.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.config,args.output)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
