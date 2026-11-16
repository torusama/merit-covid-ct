"""Prepare a validation-only Kaggle run on the exact epoch-0 source snapshot."""
import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    prior=ROOT/'outputs/kaggle_epoch0_finish'
    source=(prior/'kaggle_train.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ('BUNDLE','SOURCE_SHA')}
    sha=hashlib.sha256(base64.b64decode(constants['BUNDLE'])).hexdigest()
    provenance=json.loads((ROOT/'outputs/kaggle_review_20261116/epoch0_complete/MERIT/fold-00/source_provenance.json').read_text())
    assert sha==constants['SOURCE_SHA']==provenance['archive_sha256']
    # This prefix only unpacks the original snapshot and verifies input layout.
    # It stops BEFORE all original training/pilot/repair-test orchestration.
    prefix=source.split('\ndef run(args):',1)[0]
    assert '--epochs' not in prefix
    worker=(ROOT/'tools/validate_trained_execution.py').read_text(encoding='utf-8')
    overlay=(ROOT/'tools/merit_factorized_attention.py').read_text(encoding='utf-8')
    expected=json.loads((ROOT/'outputs/kaggle_results/source-uncompressed-hashes.json').read_text())
    runner='''
from importlib.util import spec_from_file_location, module_from_spec
validation_output=Path('/kaggle/working/trained-validation')
validation_output.mkdir(exist_ok=True)
worker_path=project/'trained_validation_worker.py'
worker_path.write_text(VALIDATION_WORKER,encoding='utf-8')
module_spec=spec_from_file_location('trained_validation_worker',worker_path)
module=module_from_spec(module_spec);module_spec.loader.exec_module(module)
checkpoints=list(input_root.rglob('checkpoints/last.pt'))
histories=list(input_root.rglob('checkpoints/history.jsonl'))
if len(checkpoints)!=1 or len(histories)!=1:raise ValueError('Expected one completed checkpoint and one history')
checkpoint=checkpoints[0]
if checkpoint.stat().st_size!=2089373860:raise ValueError('Checkpoint size differs from the verified saved epoch-0 output')
print('Verifying checkpoint bytes',flush=True)
digest=hashlib.sha256()
with checkpoint.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
(validation_output/'provenance.json').write_text(json.dumps(dict(original_source_archive_sha256=SOURCE_SHA,
    checkpoint_sha256=digest.hexdigest(),checkpoint_bytes=checkpoint.stat().st_size,
    worker_sha256=hashlib.sha256(VALIDATION_WORKER.encode()).hexdigest(),overlay_sha256=hashlib.sha256(MER_OVERLAY.encode()).hexdigest(),
    evaluation_only=True,training_updates=0),indent=2),encoding='utf-8')
result=module.run(checkpoint,manifest,histories[0],MER_OVERLAY,VALIDATION_HASHES,validation_output)
print('FINAL STATUS:',result['status'],flush=True)
'''
    script=prefix+'\nVALIDATION_WORKER = '+repr(worker)+'\nMER_OVERLAY = '+repr(overlay)+'\nVALIDATION_HASHES = '+repr(expected)+'\n'+runner
    compile(script,'kaggle_validate.py','exec')
    metadata=json.loads((prior/'kernel-metadata.json').read_text())
    metadata.update(id='vtan2405/merit-epoch0-fast-validation',title='MERIT Epoch0 Fast Execution Validation',code_file='kaggle_validate.py',kernel_sources=['vtan2405/merit-native512-fold0-epoch0-finish'])
    out=ROOT/'outputs/kaggle_trained_validation'
    out.mkdir(exist_ok=True)
    (out/'kaggle_validate.py').write_text(script,encoding='utf-8')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    (out/'package_provenance.json').write_text(json.dumps(dict(original_source_sha256=sha,
        worker_sha256=hashlib.sha256(worker.encode()).hexdigest(),overlay_sha256=hashlib.sha256(overlay.encode()).hexdigest(),
        script_sha256=hashlib.sha256(script.encode()).hexdigest(),training_updates=0),indent=2)+'\n',encoding='utf-8')
    print('Prepared validation-only kernel:',out)


if __name__=='__main__':main()
