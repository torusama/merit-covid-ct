from copy import deepcopy
import pytest
from merit_covid.engine.training import protocol_fingerprint
from scripts.kaggle.resume_checkpoint import rebind_verified_timestamps


def fixture():
    config = {'training': {'epochs': 1}, 'data': {'manifest':'/tmp/manifest.json'}}
    provenance = {'fold':0, 'source_identity': {'cases':[{'image':{'path':'ct.nii','size':8,'mtime_ns':1},
                                                        'mask':{'path':'gt.nii','size':8,'mtime_ns':1}}]}}
    cp = {'config':config, 'provenance':provenance, 'model':object(),
          'protocol_fingerprint':protocol_fingerprint(config,provenance)}
    return config, provenance, cp


def test_verified_migration_preserves_weights_and_only_rebinds_timestamps():
    cfg, prov, cp = fixture()
    current = deepcopy(prov)
    current['source_identity']['cases'][0]['image']['mtime_ns'] = 900
    result = rebind_verified_timestamps(cp,cfg,current,protocol_fingerprint(cfg,current))
    assert result['model'] is cp['model']
    assert cp['provenance'] == prov
    assert result['provenance'] == current


@pytest.mark.parametrize('key,value', [('path','redirected.nii'), ('size',9)])
def test_migration_rejects_other_source_changes(key,value):
    cfg, prov, cp = fixture()
    current = deepcopy(prov)
    current['source_identity']['cases'][0]['image'][key] = value
    with pytest.raises(ValueError,match='more than source timestamps'):
        rebind_verified_timestamps(cp,cfg,current,protocol_fingerprint(cfg,current))


def test_content_verification_rejects_changed_bytes(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from scripts.kaggle.resume_checkpoint import verify_dataset_content
    ct = tmp_path/'ct.nii'; ct.write_bytes(b'original')
    gt = tmp_path/'gt.nii'; gt.write_bytes(b'mask')
    case = SimpleNamespace(case_id='a',image=ct,mask=gt)
    expected = {'a':{'image':hashlib.sha256(b'original').hexdigest(),
                     'mask':hashlib.sha256(b'mask').hexdigest()}}
    verify_dataset_content([case],expected)
    ct.write_bytes(b'modified')
    with pytest.raises(ValueError,match='Dataset bytes differ'):
        verify_dataset_content([case],expected)
