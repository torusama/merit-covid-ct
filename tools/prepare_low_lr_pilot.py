"""Package the isolated 50-batch low-LR pilot from the frozen fast source."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    prior = ROOT/'outputs/kaggle_fast_epoch1_mkdir_fix'
    source = (prior/'kaggle_train.py').read_text(encoding='utf-8')
    prefix = source.split('\nfrom scripts.kaggle.fast_epoch1 import run as continue_epoch1', 1)[0]
    workers = {name:(ROOT/'tools'/name).read_text(encoding='utf-8')
               for name in ('epoch1_low_lr_pilot.py', 'audit_epoch1_inference.py')}
    runner = '''
(project/'tools').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
for name, worker_source in PILOT_WORKERS.items():
    (project/'tools'/name).write_text(worker_source, encoding='utf-8')
from tools.epoch1_low_lr_pilot import run as run_pilot
parents = list(input_root.rglob('checkpoints/last.pt'))
histories = list(input_root.rglob('checkpoints/history.jsonl'))
if len(parents) != 1 or len(histories) != 1:
    raise ValueError('Expected the original epoch-0 output as the only checkpoint source')
digest = hashlib.sha256()
with parents[0].open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024), b''): digest.update(block)
if digest.hexdigest() != '22596971dc0f3ff7b6796756a61cf61187ded97472a9bdf6d754766165cd6934':
    raise ValueError('Parent checkpoint bytes changed')
pilot_output = Path('/kaggle/working/low-lr-pilot')
pilot_output.mkdir(parents=True, exist_ok=True)
(pilot_output/'source.json').write_text(json.dumps(dict(source_archive_sha256=SOURCE_SHA,
    parent_checkpoint_sha256=digest.hexdigest(), worker_sha256={name:hashlib.sha256(text.encode()).hexdigest()
    for name,text in PILOT_WORKERS.items()}, learning_rate=3e-5, total_batches=50), indent=2), encoding='utf-8')
run_pilot(parents[0], histories[0], DATA_HASHES, pilot_output, total_batches=50,
    evaluate_every=10, learning_rate=3e-5,
    probe_case_ids=['volume-covid19-A-0014', 'volume-covid19-A-0319', 'volume-covid19-A-0347'])
'''
    script = prefix+'\nPILOT_WORKERS = '+repr(workers)+'\n'+runner
    ast.parse(script)
    out = ROOT/'outputs/kaggle_low_lr_pilot'; out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((prior/'kernel-metadata.json').read_text())
    meta.update(id='vtan2405/merit-low-lr-50-batch-pilot', title='MERIT Low LR 50 Batch Pilot',
                code_file='kaggle_pilot.py', kernel_sources=['vtan2405/merit-native512-fold0-epoch0-finish'])
    (out/'kaggle_pilot.py').write_text(script, encoding='utf-8')
    (out/'kernel-metadata.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    (out/'package-provenance.json').write_text(json.dumps(dict(
        source_archive_sha256=json.loads((prior/'package_provenance.json').read_text())['source_sha256'],
        script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        workers={name:hashlib.sha256(text.encode()).hexdigest() for name,text in workers.items()},
        parent_epoch=0, epoch=1, total_batches=50, evaluate_every=10,
        old_learning_rate=3e-4, new_learning_rate=3e-5,
        optimizer_reset=False, full_validation=False), indent=2)+'\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
