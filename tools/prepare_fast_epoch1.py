"""Create an immutable minimal execution-only successor to the epoch-0 snapshot."""
import ast
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unexpected frozen source: '+old[:70])
    return text.replace(old,new,1)


def make_snapshot():
    source=(ROOT/'outputs/kaggle_epoch0_finish/kaggle_train.py').read_text(encoding='utf-8')
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(source).body
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ('BUNDLE','SOURCE_SHA','EXPECTED','GIT_COMMIT')}
    raw=base64.b64decode(constants['BUNDLE'])
    assert hashlib.sha256(raw).hexdigest()==constants['SOURCE_SHA']=='edf946d8cdd39acee6169705b2f99f9cafad96123a8ae1c5344355256ef38539'
    with zipfile.ZipFile(io.BytesIO(raw)) as z:files={n:z.read(n) for n in z.namelist()}
    name='src/merit_covid/engine/training.py';text=files[name].decode().replace('\r\n','\n')
    text=replace_once(text,'activation_checkpointing=False)','activation_checkpointing=False, factorized_mer=False, mer_recomputation=True)')
    text=replace_once(text,'"augmentation_flips", "activation_checkpointing"):',
        '"augmentation_flips", "activation_checkpointing", "factorized_mer", "mer_recomputation"):')
    text=replace_once(text,'    config["training"] = settings','''    if settings["factorized_mer"] and not settings["activation_checkpointing"]:
        raise ValueError("factorized_mer requires activation checkpointing")
    if not settings["mer_recomputation"] and not settings["factorized_mer"]:
        raise ValueError("Disabling MER recomputation requires factorized MER")
    config["training"] = settings''')
    text=replace_once(text,'    # Version 1 is retained ONLY for loading historical models for inference.',
        '''    if payload["training"].get("factorized_mer") is False:
        payload["training"].pop("factorized_mer")
    if payload["training"].get("mer_recomputation") is True:
        payload["training"].pop("mer_recomputation")
    # Version 1 is retained ONLY for loading historical models for inference.''')
    text=replace_once(text,'        payload["environment"] = {',
        '        payload["execution_migration"] = getattr(self,"execution_migration",None)\n        payload["environment"] = {')
    text=replace_once(text,'        self.model.load_state_dict(checkpoint["model"])',
        '        self.execution_migration = checkpoint.get("execution_migration")\n        self.model.load_state_dict(checkpoint["model"])')
    files[name]=text.encode()
    name='src/merit_covid/engine/activation_checkpointing.py';text=files[name].decode().replace('\r\n','\n')
    text=replace_once(text,'    if training.get("amp", False):',
        '''    if training.get("factorized_mer",False):
        from .factorized_attention import enable_factorized_mer
        enable_factorized_mer(model,recompute=training.get("mer_recomputation",True))
    if training.get("amp", False):''')
    files[name]=text.encode()
    factor=(ROOT/'tools/merit_factorized_attention.py').read_text(encoding='utf-8')
    factor=replace_once(factor,'if self.training and torch.is_grad_enabled() else attention(*args)',
        'if self.training and torch.is_grad_enabled() and self._mer_recomputation else attention(*args)')
    factor=replace_once(factor,'def enable_factorized_mer(model):','def enable_factorized_mer(model, *, recompute=True):')
    factor=replace_once(factor,'    model.mer.forward = types.MethodType(factorized_mer, model.mer)',
        '    model.mer._mer_recomputation = recompute\n    model.mer.forward = types.MethodType(factorized_mer, model.mer)')
    files['src/merit_covid/engine/factorized_attention.py']=factor.encode()
    files['scripts/kaggle/fast_epoch1.py']=(ROOT/'scripts/kaggle/fast_epoch1.py').read_bytes()
    for name,content in files.items():
        if name.endswith('.py'):compile(content,name,'exec')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info=zipfile.ZipInfo(name,(2026,11,16,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,files[name])
    return buffer.getvalue(),constants


def main(output_dir=None):
    snapshot,previous=make_snapshot();sha=hashlib.sha256(snapshot).hexdigest()
    original=(ROOT/'outputs/kaggle_epoch0_finish/kaggle_train.py').read_text(encoding='utf-8')
    # Reuse the exact dataset/bootstrap procedure, replacing only archive literals.
    prefix=original.split('\ndef run(args):',1)[0]
    tree=ast.parse(prefix);lines=prefix.splitlines(keepends=True)
    replacements={'BUNDLE':base64.b64encode(snapshot).decode(),'SOURCE_SHA':sha}
    for node in reversed(tree.body):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
            key=node.targets[0].id;lines[node.lineno-1:node.end_lineno]=[key+' = '+repr(replacements[key])+'\n']
    prefix=''.join(lines)
    hashes=json.loads((ROOT/'outputs/kaggle_results/source-uncompressed-hashes.json').read_text())
    validation=json.loads((ROOT/'outputs/kaggle_trained_validation/results/trained-validation/validation.json').read_text())
    assert validation['status']=='PASS' and validation['max_absolute_metric_difference']==0 and validation['weights_unchanged']
    runner='''
from scripts.kaggle.fast_epoch1 import run as continue_epoch1
checkpoints=list(input_root.rglob('checkpoints/last.pt'))
histories=list(input_root.rglob('checkpoints/history.jsonl'))
bests=list(input_root.rglob('checkpoints/best.pt'))
if len(checkpoints)!=1 or len(histories)!=1 or len(bests)!=1:raise ValueError('Expected one original epoch-0 output source')
cp=checkpoints[0]
digest=hashlib.sha256()
with cp.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
if digest.hexdigest()!='22596971dc0f3ff7b6796756a61cf61187ded97472a9bdf6d754766165cd6934':
    raise ValueError('Unexpected checkpoint bytes')
(output/'execution-source.json').write_text(json.dumps(dict(new_source_sha256=SOURCE_SHA,
    original_source_sha256='edf946d8cdd39acee6169705b2f99f9cafad96123a8ae1c5344355256ef38539',
    parent_checkpoint_sha256=digest.hexdigest(),requested_epoch=1,stop_after_validation=True,
    training_time_limit_seconds=None,checkpoint_interval_batches=100),indent=2),encoding='utf-8')
# Preserve the notebook's explicit directory creation as well as the runner fix.
(output/'checkpoints').mkdir(parents=True,exist_ok=True)
continue_epoch1(cp,histories[0],bests[0],config,DATA_HASHES,output)
'''
    script=prefix+'\n'+runner;compile(script,'kaggle_train.py','exec')
    out=Path(output_dir) if output_dir is not None else ROOT/'outputs/kaggle_fast_epoch1'
    out.mkdir(parents=True,exist_ok=True)
    snapshot_path=out/'source_snapshot.zip'
    if snapshot_path.exists() and snapshot_path.read_bytes()!=snapshot:raise FileExistsError('Snapshot differs; use a new output directory')
    snapshot_path.write_bytes(snapshot)
    (out/'kaggle_train.py').write_text(script,encoding='utf-8')
    meta=json.loads((ROOT/'outputs/kaggle_epoch0_finish/kernel-metadata.json').read_text())
    meta.update(id='vtan2405/merit-fast-epoch1',title='MERIT Fast Epoch1',code_file='kaggle_train.py',
        kernel_sources=['vtan2405/merit-native512-fold0-epoch0-finish'])
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    (out/'package_provenance.json').write_text(json.dumps(dict(source_sha256=sha,
        original_source_sha256=previous['SOURCE_SHA'],epoch_budget=2,start_epoch=1,
        checkpoint_interval_batches=100,training_time_limit_seconds=None),indent=2)+'\n',encoding='utf-8')
    print('Prepared fast epoch-1 package',sha)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,help='New package directory; existing snapshots remain immutable')
    main(parser.parse_args().output_dir)
