"""Build a CPU Kaggle notebook with a hashed source snapshot embedded, no data."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def snapshot_files():
    files = []
    for directory in ('src', 'tools', 'scripts', 'tests', 'configs', 'splits', 'docs'):
        for path in (ROOT / directory).rglob('*'):
            if (path.is_file() and '__pycache__' not in path.parts
                    and (path.suffix in ('.py', '.md', '.yaml', '.yml', '.json', '.toml', '.txt')
                         or path.name in ('LICENSE', 'NOTICE'))):
                files.append(path)
    files += [ROOT / 'pyproject.toml', ROOT / 'README.md']
    return sorted(set(files))


def build(output, pilot_path):
    pilot_path = Path(pilot_path)
    pilot = json.loads(pilot_path.read_text(encoding='utf-8'))
    exported = json.loads((pilot_path.parent / 'b3a_precompute_manifest.json').read_text(encoding='utf-8'))
    reference = dict(case_ids=pilot['case_ids'], stats_sha256=pilot['dataset_provenance']['stats_sha256'],
        parameters=pilot['dataset_provenance']['parameters'],
        input_hashes={case['case_id']: case['source_hashes'] for case in exported['source_cases']})
    files = snapshot_files()
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr('SOURCE_SNAPSHOT.json', json.dumps(manifest, indent=2) + '\n')
    blob = stream.getvalue()
    encoded, digest = base64.b64encode(blob).decode(), hashlib.sha256(blob).hexdigest()
    def md(text):
        return dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True))
    def code(text):
        return dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=text.splitlines(keepends=True))
    cells = [md('''# B3A — Dynamic vs precomputed-volume Tiling128 (CPU)

Chọn **Accelerator: None**, thêm dataset **V3 precomputed đủ199 CT/GT** bằng Add Input,
rồi Run All. Source đầy đủ của snapshot được nhúng trong notebook: không clone Git,
không cần push code, không có CT/GT/checkpoint bên trong notebook.

Chỉ export13 TRAIN pilot, so correctness rồi đo DataLoader CPU với cùng full keys,
2orders, workers0/2, batch2 và3 repetitions sau warmup. Memory gate có thể bỏ setup2-worker
nếu RAM không đủ; báo rõ NOT_RUN. Không model/train, không GPU/VRAM, không B3B/B4/B5.

V3 dùng `native` pass-through, không chuẩn hóa lần nữa. Output mới mỗi lần chạy;
giữ giới hạn provenance, không tuyên bố199 ca raw-exact.
'''), code('''from pathlib import Path
import os, sys, subprocess, uuid
os.environ['CUDA_VISIBLE_DEVICES'] = ''
BASE = Path('/kaggle/working') / ('b3a_' + uuid.uuid4().hex[:8])
BASE.mkdir(parents=True, exist_ok=False)
PROJECT = BASE / 'project'
DATA_ROOT = None  # nếu có nhiều dataset V3, điền Path('/kaggle/input/...')
if DATA_ROOT is None:
    candidates = [p.parent for p in Path('/kaggle/input').rglob('merit_precomputed_v3.yaml')
                  if all((p.parent / n).is_file() for n in ('cases.json', 'splits_covid19_20_5fold.json', 'EXPORT_COMPLETE.json'))]
    if len(candidates) != 1:
        raise RuntimeError('Can chon dung DATA_ROOT V3; candidates=' + str(candidates))
    DATA_ROOT = candidates[0]
DATA_ROOT = Path(DATA_ROOT)
print('Data:', DATA_ROOT, '\\nOutput:', BASE)
''')]
    cells += [md('## 1. Giải nén và kiểm checksum source đã nhúng'), code(
        'import base64, hashlib, io, json, zipfile\n'
        f'SOURCE_ZIP_B64 = {encoded!r}\n'
        f'EXPECTED_SOURCE_SHA256 = {digest!r}\n'
        "payload = base64.b64decode(SOURCE_ZIP_B64)\n"
        "assert hashlib.sha256(payload).hexdigest() == EXPECTED_SOURCE_SHA256\n"
        "PROJECT.mkdir()\n"
        "with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n"
        "    assert archive.testzip() is None\n"
        "    for name in archive.namelist():\n"
        "        assert (PROJECT / name).resolve().is_relative_to(PROJECT.resolve())\n"
        "    archive.extractall(PROJECT)\n"
        "for name, expected in json.loads((PROJECT / 'SOURCE_SNAPSHOT.json').read_text()).items():\n"
        "    assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == expected, name\n"
        "print('Source SHA PASS; files:', len(json.loads((PROJECT / 'SOURCE_SNAPSHOT.json').read_text())))\n")]
    cells += [md('## 2. Dependency và test synthetic trước khi đọc pilot thật'), code('''import importlib.util
small_dependencies = {'numpy':'numpy>=1.24', 'nibabel':'nibabel>=5', 'yaml':'PyYAML>=6',
                      'psutil':'psutil>=5', 'pytest':'pytest>=7'}
missing = [pkg for module, pkg in small_dependencies.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
if importlib.util.find_spec('torch') is None:
    raise RuntimeError('Can cai Torch CPU truoc; notebook khong tu tai bo Torch/CUDA lon.')
XML = BASE / 'synthetic.xml'
test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_b3a_delivery.py', '-q',
    '--basetemp', str(BASE / 'pytest_tmp'), '--junitxml', str(XML)], cwd=PROJECT,
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(BASE / 'synthetic.log').write_text(test.stdout, encoding='utf-8')
print(test.stdout)
test.check_returncode()
'''), md('## 3. Stage cùng storage, export subset, correctness A-0003 rồi benchmark\n\nLượt này có thể mất nhiều phút. Log mỗi2000 batch; không phải train. Không dừng trước correctness để tự lấy số benchmark.'), code('''EXPERIMENT = BASE / 'experiment'
command = [sys.executable, '-u', 'tools/b3a_compare.py', '--data-root', str(DATA_ROOT),
    '--output-dir', str(EXPERIMENT), '--synthetic-xml', str(XML),
    '--workers', '0', '2', '--batch-size', '2', '--repetitions', '3']
with (BASE / 'benchmark.log').open('w', encoding='utf-8') as log:
    process = subprocess.Popen(command, cwd=PROJECT, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, bufsize=1)
    for line in process.stdout:
        print(line, end='')
        log.write(line); log.flush()
    if process.wait() != 0:
        raise RuntimeError('B3A STOP; xem benchmark.log/B3A_ERROR.json, khong tiep tuc model/train.')
'''), md('## 4. Báo cáo và ZIP nhỏ để gửi review'), code('''REPORT = BASE / 'report'
subprocess.run([sys.executable, 'tools/b3a_reports.py', '--experiment', str(EXPERIMENT),
                '--output-dir', str(REPORT)], cwd=PROJECT, check=True)
print((REPORT / 'b3a_decision.md').read_text(encoding='utf-8'))
import zipfile
REPORT_ZIP = BASE / 'B3A_KAGGLE_REPORT.zip'
with zipfile.ZipFile(REPORT_ZIP, 'x', zipfile.ZIP_DEFLATED) as archive:
    for p in EXPERIMENT.iterdir():
        if p.is_file() and p.suffix in ('.json', '.csv'):
            archive.write(p, 'reports/' + p.name)
    for p in REPORT.iterdir():
        if p.is_file(): archive.write(p, p.name)
    for p in (XML, BASE/'synthetic.log', BASE/'benchmark.log'):
        archive.write(p, 'evidence/' + p.name)
    archive.write(PROJECT/'SOURCE_SNAPSHOT.json', 'SOURCE_SNAPSHOT.json')
print('Tai ZIP tu Output Files:', REPORT_ZIP)
print('Da dung B3A; khong B3B/model/train/GPU benchmark.')
''')]
    for index, cell in enumerate(cells):
        cell['id'] = f'b3a-{index:02d}'
    benchmark_cell = next(cell for cell in cells if cell['cell_type'] == 'code'
                          and ''.join(cell['source']).startswith('EXPERIMENT ='))
    reference_check = ('import json\n'
        f'EXPECTED_PILOT = json.loads({json.dumps(reference)!r})\n'
        "sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')]\n"
        "from tools.qc_tiling128 import load_inputs\n"
        "from tools.b3a_compare import select_pilot\n"
        "cases, parts, config, exported, entries, _, _ = load_inputs(DATA_ROOT, 'cases.json', 'splits_covid19_20_5fold.json', 'merit_precomputed_v3.yaml', 'EXPORT_COMPLETE.json')\n"
        "chosen = select_pilot(cases, entries, parts)\n"
        "assert [c.case_id for c in chosen] == EXPECTED_PILOT['case_ids'], 'Pilot subset differs from reviewed local experiment'\n"
        "assert exported['stats_sha256'] == EXPECTED_PILOT['stats_sha256'] and exported['parameters'] == EXPECTED_PILOT['parameters'], 'Different V3 stats'\n"
        "for name, expected in EXPECTED_PILOT['input_hashes'].items():\n"
        "    assert entries[name]['ct_output_sha256'] == expected['ct'] and entries[name]['gt_output_sha256'] == expected['gt'], name\n"
        "print('Pinned13-case subset and V3 metadata hashes match local reference.')\n")
    benchmark_cell['source'] = (reference_check + ''.join(benchmark_cell['source'])).splitlines(keepends=True)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                      language_info=dict(name='python'), b3a_source_sha256=digest), cells=cells)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(json.dumps(dict(notebook=str(output), embedded_source_files=len(files),
                         embedded_source_zip_bytes=len(blob), embedded_source_sha256=digest)))
    return notebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True, help='Reviewed local b3a_pilot.json; pins13 IDs/source hashes')
    args = parser.parse_args()
    build(args.output, args.pilot)
