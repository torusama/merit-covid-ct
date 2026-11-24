"""Small CPU tests of handoff QC; no model/metric/fit execution."""
from pathlib import Path
from dataclasses import asdict
import json

import nibabel as nib
import numpy as np
import pytest
import yaml

from scripts.qc_preprocessing import qc_case,run
from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.config import PreprocessConfig

ROOT=Path(__file__).resolve().parents[1]


def config():
    stats=json.loads((ROOT/'analysis/preprocessing/fit_stats.json').read_text())
    return PreprocessConfig(intensity_encoding=stats['policy'],zscore_fit=stats,
        zscore_mean=stats['parameters']['mean'],zscore_std=stats['parameters']['std'],assume_mm=True)


def pair(tmp_path,*,bad_label=False,bad_affine=False):
    image=np.zeros((512,512,3),dtype=np.int16)-1024
    mask=np.zeros(image.shape,dtype=np.float64)
    mask[5,8,0]=1+1e-10 if bad_label else 1
    mask[400,30,2]=1
    affine=np.diag([-1.,-1.,-2.,1.])
    mask_affine=affine.copy()
    if bad_affine:mask_affine[0,3]=7
    a,b=tmp_path/'synthetic_ct.nii',tmp_path/'synthetic_seg.nii'
    for values,path,matrix in ((image,a,affine),(mask,b,mask_affine)):
        nii=nib.Nifti1Image(values,matrix);nii.header.set_xyzt_units('mm');nib.save(nii,path)
    return Case('synthetic',a,b,'synthetic')


def test_qc_exact_lpi_roundtrip_and_hashes(tmp_path):
    row=qc_case(pair(tmp_path),config(),'train')
    assert row['status']=='PASS'
    assert row['gt_voxels_before']==row['gt_voxels_canonical']==row['gt_voxels_restored']==2
    assert row['gt_coordinate_mismatches']==0
    assert row['canonical_orientation']==['L','P','S']
    assert row['ct_sha256_before']==row['ct_sha256_after']
    assert row['gt_sha256_before']==row['gt_sha256_after']


@pytest.mark.parametrize('invalid',['label','affine'])
def test_qc_does_not_repair_bad_gt(tmp_path,invalid):
    row=qc_case(pair(tmp_path,bad_label=invalid=='label',bad_affine=invalid=='affine'),config(),'test')
    assert row['status']=='FAIL' and row['reason']


def test_missing_pair_is_blocked(tmp_path):
    row=qc_case(Case('missing',tmp_path/'missing_ct.nii',None,'missing'),config(),'validation')
    assert row['status']=='BLOCKED_MISSING_DATA'


def test_wrong_filename_case_identity_rejected(tmp_path):
    c=pair(tmp_path)
    row=qc_case(Case('different',c.image,c.mask,'different'),config(),'train')
    assert row['status']=='FAIL' and 'case ID' in row['reason']


def test_output_overwrite_refused(tmp_path):
    with pytest.raises(FileExistsError):run(ROOT/'configs/merit_preprocessing.yaml',tmp_path)


def test_missing_cohort_does_not_claim_pass(tmp_path):
    # Preserve exact provided manifest/split/stats bytes; move metadata only to
    # an empty folder so all CT paths are missing. No real CT is opened.
    manifest=tmp_path/'cases.json'
    manifest.write_bytes((ROOT/'tests/fixtures/covid19_20_cases_metadata.json').read_bytes())
    raw=yaml.safe_load((ROOT/'configs/merit_preprocessing.yaml').read_text())
    raw['data']['manifest']=str(manifest)
    raw['data']['split_path']=str(ROOT/'splits/covid19_20_5fold.json')
    raw['preprocess']['zscore_fit']=str(ROOT/'analysis/preprocessing/fit_stats.json')
    cfg=tmp_path/'config.yaml';cfg.write_text(yaml.safe_dump(raw))
    result=run(cfg,tmp_path/'qc')
    assert result['status']=='BLOCKED_MISSING_DATA' and result['qc_completed']==0
