"""Package a bounded diagnostic using the deployed epoch-1 source unchanged."""
import ast
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(gradient_only=False):
    prior = ROOT/'outputs/kaggle_fast_epoch1_mkdir_fix'
    source = (prior/'kaggle_train.py').read_text(encoding='utf-8')
    prefix = source.split('\nfrom scripts.kaggle.fast_epoch1 import run as continue_epoch1', 1)[0]
    worker = (ROOT/'tools/audit_epoch1_inference.py').read_text(encoding='utf-8')
    runner = '''
from importlib.util import spec_from_file_location, module_from_spec
worker_path = project/'audit_epoch1_inference.py'
worker_path.write_text(AUDIT_WORKER, encoding='utf-8')
spec = spec_from_file_location('audit_epoch1_inference', worker_path)
worker = module_from_spec(spec); spec.loader.exec_module(worker)
best = list(input_root.rglob('checkpoints/best.pt'))
last = list(input_root.rglob('checkpoints/last.pt'))
if len(best) != 1 or len(last) != 1:
    raise ValueError('Expected epoch-1 output as the only checkpoint source')
worker.run(best[0], last[0], manifest, Path('/kaggle/working/epoch1-quality-audit'))
'''
    if gradient_only:
        runner = runner.replace('worker.run(', 'worker.run_positive_gradient(')
    script = prefix + '\nAUDIT_WORKER = ' + repr(worker) + '\n' + runner
    ast.parse(script)
    meta = json.loads((prior/'kernel-metadata.json').read_text())
    meta.update(id='vtan2405/merit-epoch1-quality-audit', title='MERIT Epoch1 Quality Audit',
                code_file='kaggle_audit.py', kernel_sources=['vtan2405/merit-fast-epoch1'])
    out = ROOT/'outputs/kaggle_epoch1_quality_audit'/('positive_gradient_package' if gradient_only else 'package')
    out.mkdir(parents=True, exist_ok=True)
    (out/'kaggle_audit.py').write_text(script, encoding='utf-8')
    (out/'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps({'worker_sha256': hashlib.sha256(worker.encode()).hexdigest(),
        'source_archive_sha256': json.loads((prior/'package_provenance.json').read_text())['source_sha256'],
        'training_updates': 0, 'diagnostic_slices_selected_using_labels': True,
        'gradient_only': gradient_only}, indent=2) + '\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gradient-only', action='store_true')
    main(parser.parse_args().gradient_only)
