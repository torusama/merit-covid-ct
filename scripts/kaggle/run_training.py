import base64, gzip, hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
started = time.monotonic()
setup_only = os.environ.get('MERIT_SETUP_ONLY') == '1'
project = Path(os.environ.get('MERIT_PROJECT_DIR', '/tmp/merit-project')).resolve()
project.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(project)
os.chdir(project)
sys.path[:0] = [str(project / 'src'), str(project)]
os.environ['PYTHONPATH'] = os.pathsep.join([str(project / 'src'), str(project), os.environ.get('PYTHONPATH', '')])
print('Source archive SHA256:', SOURCE_SHA, flush=True)
if os.environ.get('MERIT_SKIP_INSTALL') != '1':
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[unetrpp]'], check=True)
import merit_covid
if Path(merit_covid.__file__).resolve() != project / 'src/merit_covid/__init__.py':
    raise RuntimeError('Imported merit_covid from outside the packaged source')
print('Package loaded from:', merit_covid.__file__, flush=True)
import torch
if not setup_only:
    assert torch.cuda.is_available(), 'Kaggle GPU unavailable'
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
input_root = Path(os.environ.get('MERIT_INPUT_DIR', '/kaggle/input'))
from scripts.kaggle.dataset_layout import resolve_case_files, case_paths
roots = [p for p in input_root.rglob('Train') if p.is_dir()
         and any((p / name).is_file() for name in case_paths('volume-covid19-A-0025', 'seg'))]
if len(roots) != 1:
    raise RuntimeError(f'Expected one Kaggle dataset root, found {roots}')
train = roots[0]
manifest = project / 'data/covid19_20/cases.json'
records = json.loads(manifest.read_text(encoding='utf-8-sig'))
sys.path.insert(0, str(project))
available = {p.relative_to(train).as_posix() for p in train.rglob('*') if p.is_file()}
layout_errors = []
for row in records:
    case = row['id']
    try:
        paths = resolve_case_files(train, case, available)
        row['image'], row['mask'] = str(paths['image']), str(paths['mask'])
    except ValueError as error:
        layout_errors.append(str(error))
if layout_errors:
    raise FileNotFoundError('Dataset layout errors:\n' + '\n'.join(layout_errors))
pilot_paths = resolve_case_files(train, 'volume-covid19-A-0025', available)
for kind, path in [('ct', pilot_paths['image']), ('seg', pilot_paths['mask'])]:
    h = hashlib.sha256()
    with (gzip.open(path, 'rb') if path.name.endswith('.gz') else path.open('rb')) as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    if h.hexdigest() != EXPECTED[kind]: raise ValueError(f'A-0025 {kind} differs from local source')
print('Dataset: 199 paths present; A-0025 CT and mask byte hashes match local decompressed data', flush=True)
manifest.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
from merit_covid.data.pipeline.cases import load_cases, manifest_fingerprint
from merit_covid.data.pipeline.splits import validate_splits
cases, _ = load_cases(manifest, expected_count=199)
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.geometry import inspect_pair
by_id = {row['id']: row for row in records}
for case in cases:
    ct, mask = inspect_pair(case, PreprocessConfig(assume_mm=True))
    if list(ct.shape) != by_id[case.case_id]['shape']:
        raise ValueError(f'{case.case_id}: CT header shape differs from local manifest')
print('All 199 CT/mask header pairs passed shape, affine, spacing and orientation checks', flush=True)
split_path = project / 'splits/covid19_20_5fold.json'
split = json.loads(split_path.read_text(encoding='utf-8-sig'))
old_fingerprint = split['manifest_fingerprint']
split['manifest_fingerprint'] = manifest_fingerprint(cases)
validate_splits(split, cases)
split_path.write_text(json.dumps(split, indent=2) + '\n', encoding='utf-8')
print('Split IDs unchanged; filename fingerprint adjusted for Kaggle layout:', old_fingerprint, '->', split['manifest_fingerprint'], flush=True)
config = project / 'configs/merit_native512.yaml'
output = Path(os.environ.get('MERIT_OUTPUT_DIR', '/kaggle/working/MERIT/fold-00'))
output.mkdir(parents=True, exist_ok=True)
(output / 'source_provenance.json').write_text(json.dumps({'archive_sha256': SOURCE_SHA, 'git_commit': GIT_COMMIT,
  'local_worktree_changes': True, 'dataset': 'vtan2405/merit-covid19-20', 'fold': 0,
  'original_split_fingerprint': old_fingerprint, 'kaggle_split_fingerprint': split['manifest_fingerprint'],
  'a0025_uncompressed_sha256': EXPECTED, 'assume_mm': True}, indent=2), encoding='utf-8')
if setup_only:
    from merit_covid.engine.training import load_experiment, split_datasets
    datasets, provenance = split_datasets(load_experiment(config), 0)
    print('Setup verified; slab counts:', {k: len(v) for k, v in datasets.items()}, flush=True)
    raise SystemExit(0)
def run(args):
    print('RUN:', ' '.join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], cwd=project, check=True)
pilot = Path('/tmp/merit-pilot')
if not RESUME_REQUIRED:
    run([sys.executable, '-u', 'scripts/merit_preflight.py', 'pilot', '--config', config,
         '--fold', '0', '--case-id', 'volume-covid19-A-0025', '--size', '512', '--steps', '2',
         '--device', 'cuda', '--output', pilot])
    shutil.copy2(pilot / 'pilot.json', output / 'pilot-report.json')
    shutil.rmtree(pilot)
checkpoint_dir = output / 'checkpoints'
args = [sys.executable, '-u', 'scripts/train_colab.py', '--config', config, '--fold', '0',
        '--output', checkpoint_dir, '--epochs', '1', '--checkpoint-every', '25']
if RESUME_REQUIRED:
    candidates = list(Path('/kaggle/input').glob('**/in_progress.pt'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one prior in_progress.pt; found {candidates}')
    args += ['--resume', candidates[0]]
    hashes_path = project / 'verified-source-hashes.json'
    hashes_path.write_text(json.dumps(DATA_HASHES), encoding='utf-8')
    args += ['--verified-resume-hashes', hashes_path]
    print('Resuming trusted checkpoint:', candidates[0], flush=True)
    (output/'recovery-policy.json').write_text(json.dumps({'source_archive':REPAIR_SOURCE_SHA,
        'policy':'verify all CT/mask SHA256; rebind timestamps only; retry nonfinite AMP Conv3d in FP32',
        'resume_from':str(candidates[0])}, indent=2), encoding='utf-8')
# Reserve several hours of the 12-hour session for validation and output persistence.
remaining = 7 * 3600 - (time.monotonic() - started)
if remaining <= 0:
    raise RuntimeError('Pilot used the training time budget; no training was started')
args += ['--max-seconds', str(remaining)]
try:
    run(args)
    status = 'epoch_complete' if (checkpoint_dir / 'last.pt').is_file() else 'paused_at_checkpoint'
except subprocess.CalledProcessError as error:
    status = 'training_error'
    print('Training stopped with an error; the latest completed checkpoint is retained:', error, flush=True)
(output / 'run-status.json').write_text(json.dumps({'status': status,
    'checkpoint': 'checkpoints/last.pt' if status == 'epoch_complete' else
                  'checkpoints/in_progress.pt' if (checkpoint_dir / 'in_progress.pt').is_file() else None}, indent=2), encoding='utf-8')
print('Run status:', status, flush=True)
