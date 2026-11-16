"""Package the current training source for a private Kaggle kernel."""
import base64
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--resume-from', help='Kaggle kernel with a saved in_progress.pt output')
mode.add_argument('--repair-from', help='Verified recovery of the known version-4 numerical failure')
args = parser.parse_args()
resume_source = args.resume_from or args.repair_from
DEST = ROOT / ('outputs/kaggle_recovery' if args.repair_from else 'outputs/kaggle_native512_fold0')
DEST.mkdir(parents=True, exist_ok=True)
bundle = io.BytesIO()
with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for directory in ('src', 'scripts', 'configs', 'splits'):
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                archive.write(path, path.relative_to(ROOT).as_posix())
    for name in ('pyproject.toml', 'data/covid19_20/cases.json'):
        archive.write(ROOT / name, name)

hashes = {}
for kind in ('ct', 'seg'):
    source = ROOT / f'data/covid19_20/Train/volume-covid19-A-0025_{kind}.nii.gz'
    digest = hashlib.sha256()
    with gzip.open(source, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    hashes[kind] = digest.hexdigest()

code = (ROOT / 'scripts/kaggle/run_training.py').read_text(encoding='utf-8')
import subprocess
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
archive = bundle.getvalue()
previous = None
if resume_source:
    # Resume only with the exact code and config snapshot that produced the checkpoint.
    with tempfile.TemporaryDirectory(prefix='merit_kaggle_provenance_') as directory:
        subprocess.run(['kaggle', 'kernels', 'output', resume_source, '-p', directory,
                        '--file-pattern', 'source_provenance\\.json$'], check=True)
        reports = list(Path(directory).rglob('source_provenance.json'))
        if len(reports) != 1:
            raise RuntimeError('Saved source provenance is missing or ambiguous; cannot verify resume code')
        previous = json.loads(reports[0].read_text(encoding='utf-8'))
        if args.repair_from and previous.get('archive_sha256') != 'c4935b667dcf824c406a0bcc571ac097b9994489c95f65c271e54625322043f6':
            raise ValueError('Repair is limited to the diagnosed version-4 source archive')
        if not args.repair_from and previous.get('archive_sha256') != hashlib.sha256(archive).hexdigest():
            raise ValueError('Current source archive differs from the saved run; refusing unsafe resume')
data_hashes = (json.loads((ROOT/'outputs/kaggle_results/source-uncompressed-hashes.json').read_text())
               if resume_source else None)
prefix = ('BUNDLE = ' + repr(base64.b64encode(archive).decode()) + '\n'
          + 'SOURCE_SHA = ' + repr(hashlib.sha256(archive).hexdigest()) + '\n'
          + 'GIT_COMMIT = ' + repr(commit) + '\nEXPECTED = ' + repr(hashes) + '\n'
          + 'RESUME_REQUIRED = ' + repr(bool(resume_source)) + '\n'
          + 'DATA_HASHES = ' + repr(data_hashes) + '\n'
          + 'REPAIR_SOURCE_SHA = ' + repr(previous.get('archive_sha256') if args.repair_from else None) + '\n')
script = prefix + code
compile(script, 'kaggle_train.py', 'exec')
(DEST / 'kaggle_train.py').write_text(script, encoding='utf-8')
metadata = {'id': 'vtan2405/merit-native512-fold0-resume' if args.resume_from else 'vtan2405/merit-native512-fold0',
            'title': 'MERIT Native512 Fold0 Resume' if args.resume_from else 'MERIT Native512 Fold0',
            'code_file': 'kaggle_train.py', 'language': 'python', 'kernel_type': 'script',
            'is_private': 'true', 'enable_gpu': 'true', 'enable_internet': 'true',
            'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': ['vtan2405/merit-covid19-20'],
            'competition_sources': [], 'kernel_sources': [resume_source] if resume_source else [], 'model_sources': []}
if args.repair_from:
    metadata.update(id='vtan2405/merit-native512-fold0-recovery', title='MERIT Native512 Fold0 Recovery')
(DEST / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
print('Prepared', DEST, 'source SHA256', hashlib.sha256(archive).hexdigest(), 'bytes', len(archive))
