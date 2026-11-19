import json
from pathlib import Path
import struct
import sys
from types import ModuleType
import zipfile

import torch

from tools.recover_colab_checkpoints import recover_checkpoints, install_recovered_reader


def snapshot(path, arm, *, binding='binding', steps=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema_version='schema', input_binding=binding, signature=arm,
                    report=dict(arm=arm, processed_batches=steps, requested_batches=300),
                    model=dict(weight=torch.ones(2))), path)


def test_recovers_both_local_saves_preserves_corrupt_drive_and_verifies_backup(tmp_path):
    work=tmp_path/'local';output=tmp_path/'drive'
    originals={}
    for arm in ('reference_alignment','alignment_off'):
        snapshot(work/(arm+'-save.pt'),arm)
        dest=output/arm;dest.mkdir(parents=True)
        (dest/'progress.pt').write_bytes(b'PK\x03\x04truncated')
        (dest/'arm-report.json').write_text(json.dumps(dict(signature=arm)))
        originals[arm]=(dest/'progress.pt').read_bytes()
    recovered,receipt=recover_checkpoints(dict(work=str(work)),output,
        expected_binding='binding',schema='schema')
    assert not receipt['failures'] and len(recovered)==2
    for arm,row in receipt['recovered'].items():
        assert (output/arm/'progress.pt').read_bytes()==originals[arm]
        assert Path(row['recovery_copy']).read_bytes()==Path(row['local']).read_bytes()
    namespace=dict(local_snapshot=lambda context,path:'original-reader')
    exec('def validate(): pass',namespace)
    install_recovered_reader(namespace['validate'],recovered)
    assert namespace['local_snapshot']({},output/'reference_alignment/progress.pt')==work/'reference_alignment-save.pt'
    assert namespace['local_snapshot']({},output/'other/progress.pt')=='original-reader'


def test_rejects_wrong_identity_and_incomplete_snapshot_then_uses_drive_copy(tmp_path):
    work=tmp_path/'local';output=tmp_path/'drive'
    arm='reference_alignment'
    snapshot(work/(arm+'-save.pt'),arm,binding='wrong')
    snapshot(work/(arm+'-read-old.pt'),arm,steps=250)
    snapshot(output/arm/'progress.pt',arm)
    recovered,receipt=recover_checkpoints(dict(work=str(work)),output,
        expected_binding='binding',schema='schema',persist_recovery=False)
    assert len(recovered)==1
    assert len(receipt['recovered'][arm]['rejected'])==2
    assert 'alignment_off' in receipt['failures']
    chosen=Path(next(iter(recovered.values())))
    assert chosen.parent==work and '-recovery-' in chosen.name


def test_reports_truncated_files_without_overwriting_them(tmp_path):
    work=tmp_path/'local';output=tmp_path/'drive'
    arm='reference_alignment';dest=output/arm;dest.mkdir(parents=True)
    broken=dest/'progress.pt';broken.write_bytes(b'PK\x03\x04no central directory')
    recovered,receipt=recover_checkpoints(dict(work=str(work)),output,
        expected_binding='binding',schema='schema')
    assert not recovered and len(receipt['failures'])==2
    assert broken.read_bytes()==b'PK\x03\x04no central directory'
    assert not list(work.glob('*-recovery-*.pt'))


def test_rejects_damaged_tensor_even_when_zip_directory_is_intact(tmp_path):
    work=tmp_path/'local';output=tmp_path/'drive';arm='reference_alignment'
    local=work/(arm+'-save.pt')
    snapshot(local,arm)
    snapshot(output/arm/'progress.pt',arm)
    with zipfile.ZipFile(local) as archive:
        entry=next(info for info in archive.infolist() if info.filename.endswith('/data/0'))
    with local.open('r+b') as stream:
        stream.seek(entry.header_offset+26)
        name_length,extra_length=struct.unpack('<HH',stream.read(4))
        offset=entry.header_offset+30+name_length+extra_length
        stream.seek(offset);value=stream.read(1)
        stream.seek(offset);stream.write(bytes([value[0]^1]))
    assert zipfile.is_zipfile(local)
    recovered,receipt=recover_checkpoints(dict(work=str(work)),output,
        expected_binding='binding',schema='schema',persist_recovery=False)
    assert len(recovered)==1
    assert 'CRC' in receipt['recovered'][arm]['rejected'][0]['error']


def test_embedded_notebook_cells_recover_then_validate_both_without_training(tmp_path, monkeypatch):
    from tools.prepare_colab_alignment_notebook import validation_cell_source
    work=tmp_path/'local';output=tmp_path/'drive'
    for arm in ('reference_alignment','alignment_off'):
        snapshot(work/(arm+'-save.pt'),arm)
        folder=output/arm;folder.mkdir(parents=True)
        (folder/'progress.pt').write_bytes(b'PK\x03\x04broken')
    state_module=ModuleType('tools.order_optimization_pilot')
    state_module.state_hash=lambda value:'binding'
    monkeypatch.setitem(sys.modules,state_module.__name__,state_module)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    namespace=dict(context=dict(work=str(work),parent_sha256='p',source_sha256='s',hashes={}),
        OUTPUT_DIR=output,TOTAL_BATCHES=300,RUN_FULL_VALIDATION=True,
        SESSION_STARTED=0,SESSION_BUDGET_MINUTES=90,remaining_seconds=lambda:100,
        SCHEMA='schema',torch=torch,Path=Path,calls=[],
        local_snapshot=lambda context,path:path)
    exec('''def validate_arm(context, output, arm, **kwargs):
    checkpoint=torch.load(local_snapshot(context, output/arm/'progress.pt'), map_location='cpu', weights_only=False)
    assert checkpoint['report']['processed_batches']==300
    calls.append(arm)
    return {'status':'validation_complete'}
''',namespace)
    exec(validation_cell_source('reference_alignment'),namespace)
    exec(validation_cell_source('alignment_off'),namespace)
    assert namespace['calls']==['reference_alignment','alignment_off']
    assert namespace['SESSION_STARTED']>0
    for arm in namespace['calls']:
        assert (output/arm/'progress.pt').read_bytes()==b'PK\x03\x04broken'


def test_embedded_notebook_recovery_respects_validation_disabled(tmp_path, monkeypatch):
    from tools.prepare_colab_alignment_notebook import validation_cell_source
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    namespace=dict(RUN_FULL_VALIDATION=False,torch=torch)
    exec(validation_cell_source('reference_alignment'),namespace)
    exec(validation_cell_source('alignment_off'),namespace)
    assert 'recovered_paths' not in namespace
