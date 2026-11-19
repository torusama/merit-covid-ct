"""Package a short five-arm BatchNorm/case-order audit using frozen fast source."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    prior = ROOT/'outputs/kaggle_fast_epoch1_mkdir_fix'
    source = (prior/'kaggle_train.py').read_text(encoding='utf-8')
    prefix = source.split('\nfrom scripts.kaggle.fast_epoch1 import run as continue_epoch1', 1)[0]
    workers = {name:(ROOT/'tools'/name).read_text(encoding='utf-8') for name in
               ('audit_bn_volume_pilot.py', 'epoch1_low_lr_pilot.py', 'audit_epoch1_inference.py')}
    previous = json.loads((ROOT/'outputs/kaggle_low_lr_pilot/results_v1/low-lr-pilot/pilot-report.json').read_text())
    expected = [[row['case'], row['master']] for row in previous['train'][:10]]
    runner = '''
(project/'tools').mkdir(parents=True, exist_ok=True)
for name, text in AUDIT_WORKERS.items():
    (project/'tools'/name).write_text(text, encoding='utf-8')
from tools.audit_bn_volume_pilot import run as run_audit
parents = list(input_root.rglob('checkpoints/last.pt'))
histories = list(input_root.rglob('checkpoints/history.jsonl'))
if len(parents) != 1 or len(histories) != 1:
    raise ValueError('Expected original epoch-0 output as the only parent source')
digest = hashlib.sha256()
with parents[0].open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024), b''): digest.update(block)
if digest.hexdigest() != '22596971dc0f3ff7b6796756a61cf61187ded97472a9bdf6d754766165cd6934':
    raise ValueError('Parent checkpoint bytes changed')
audit_output = Path('/kaggle/working/bn-volume-audit')
audit_output.mkdir(parents=True, exist_ok=True)
(audit_output/'source.json').write_text(json.dumps(dict(source_archive_sha256=SOURCE_SHA,
    parent_checkpoint_sha256=digest.hexdigest(), workers={name:hashlib.sha256(text.encode()).hexdigest()
    for name,text in AUDIT_WORKERS.items()}, per_case_batches=10, learning_rate=3e-5), indent=2), encoding='utf-8')
run_audit(parents[0], histories[0], DATA_HASHES, audit_output, per_case=10,
    learning_rate=3e-5, expected_first_batches=EXPECTED_FIRST_BATCHES,
    probe_case_ids=['volume-covid19-A-0014', 'volume-covid19-A-0319', 'volume-covid19-A-0347'])
'''
    script = prefix+'\nAUDIT_WORKERS = '+repr(workers)+'\nEXPECTED_FIRST_BATCHES = '+repr(expected)+'\n'+runner
    ast.parse(script)
    out = ROOT/'outputs/kaggle_bn_volume_audit'; out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((prior/'kernel-metadata.json').read_text())
    meta.update(id='vtan2405/merit-bn-volume-audit', title='MERIT BN Volume Audit',
                code_file='kaggle_audit.py', kernel_sources=['vtan2405/merit-native512-fold0-epoch0-finish'])
    (out/'kaggle_audit.py').write_text(script, encoding='utf-8')
    (out/'kernel-metadata.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    (out/'package-provenance.json').write_text(json.dumps(dict(
        source_archive_sha256=json.loads((prior/'package_provenance.json').read_text())['source_sha256'],
        script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        workers={name:hashlib.sha256(text.encode()).hexdigest() for name,text in workers.items()},
        parent_epoch=0, per_case_batches=10, learning_rate=3e-5,
        expected_reference_batch_order=expected, arms=5, total_optimizer_updates=20,
        full_validation=False, saves_production_checkpoint=False), indent=2)+'\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
