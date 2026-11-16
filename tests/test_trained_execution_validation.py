"""The validation gate must catch case swaps and per-case regression hidden by a mean."""
from copy import deepcopy

import pytest

from tools.validate_trained_execution import compare_reports


def fixture():
    return dict(per_case=[dict(case_id=c,group_id=c,ground_truth_foreground_voxels=10,
        dice=.4,iou=.25,precision=.5,recall=.5) for c in ('a','b')],
        macro={k:dict(mean=v) for k,v in [('dice',.4),('iou',.25),('precision',.5),('recall',.5)]})


def test_equal_reports_pass():
    baseline=fixture()
    assert compare_reports(deepcopy(baseline),baseline)['status']=='PASS'


def test_cancelling_case_regressions_are_detected():
    baseline=fixture();candidate=deepcopy(baseline)
    candidate['per_case'][0]['dice']+=.01;candidate['per_case'][1]['dice']-=.01
    assert compare_reports(candidate,baseline)['status']=='REVIEW_REQUIRED'


@pytest.mark.parametrize('change',['missing','duplicate','group','truth'])
def test_identity_and_label_mismatches_rejected(change):
    baseline=fixture();candidate=deepcopy(baseline)
    if change=='missing':candidate['per_case'].pop()
    elif change=='duplicate':candidate['per_case'].append(candidate['per_case'][0])
    elif change=='group':candidate['per_case'][0]['group_id']='wrong'
    else:candidate['per_case'][0]['ground_truth_foreground_voxels']=99
    with pytest.raises(ValueError):compare_reports(candidate,baseline)
