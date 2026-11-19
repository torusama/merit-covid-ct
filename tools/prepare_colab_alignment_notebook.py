"""Generate a self-contained Colab notebook from reviewed frozen source."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/colab_alignment_pair'
NOTEBOOK=ROOT/'notebooks/colab_alignment_pair.ipynb'
CELL_SCRIPT=ROOT/'notebooks/colab_alignment_pair_cells.py'


def drive_mount_cell_source():
    return """import os, time, re, shutil
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
""" + (ROOT/'tools/colab_drive_paths.py').read_text(encoding='utf-8') + """
from google.colab import drive
mounted_my_drive = ensure_drive_mount(
    drive, "/content/drive", force=globals().get("FORCE_REMOUNT_DRIVE", False))
print("My Drive:", mounted_my_drive)
""" + drive_project_setup_source(include_helpers=False)


def drive_project_setup_source(include_helpers=True):
    helpers = (ROOT/'tools/colab_drive_paths.py').read_text(encoding='utf-8') if include_helpers else ''
    return helpers + """
previous_drive_root = DRIVE_ROOT
DRIVE_ROOT = resolve_drive_project(
    DRIVE_ROOT, globals().get("SHARED_PROJECT_PATH", ""), globals().get("SHARED_FOLDER_URL", ""),
    my_drive=mounted_my_drive,
)
DATASET_SOURCE = rebase_drive_input(DATASET_SOURCE, previous_drive_root, DRIVE_ROOT)
EPOCH0_CHECKPOINT = rebase_drive_input(EPOCH0_CHECKPOINT, previous_drive_root, DRIVE_ROOT)
EPOCH0_HISTORY = rebase_drive_input(EPOCH0_HISTORY, previous_drive_root, DRIVE_ROOT)
print("Project folder/shortcut:", DRIVE_ROOT)
print("Shared folder link:", globals().get("SHARED_FOLDER_URL", ""))
"""


def validation_cell_source(arm):
    """Embed recovery outside the frozen training bundle to keep run identity."""
    if arm not in ('reference_alignment', 'alignment_off'):
        raise ValueError('Unknown alignment arm')
    prepare = ''
    if arm == 'reference_alignment':
        prepare = (ROOT/'tools/recover_colab_checkpoints.py').read_text(encoding='utf-8') + """
import time
if RUN_FULL_VALIDATION:
    if "context" not in globals() or "validate_arm" not in globals():
        raise RuntimeError("Chạy Cells 1–7 với cùng RUN_NAME trước; không cần chạy lại Cells 8–9.")
    from tools.order_optimization_pilot import state_hash
    expected_binding = state_hash({key: context[key] for key in ("parent_sha256", "source_sha256", "hashes")})
    recovered_paths, recovery_review = recover_checkpoints(
        context, OUTPUT_DIR, expected_binding=expected_binding,
        schema=validate_arm.__globals__["SCHEMA"], total_batches=TOTAL_BATCHES,
    )
    if not recovered_paths:
        raise RuntimeError("Không tìm thấy checkpoint nguyên vẹn, đúng dữ liệu và đủ 300 batch. Giữ output kiểm tra; chưa train đè.")
    install_recovered_reader(validate_arm, recovered_paths)
    SESSION_STARTED = time.monotonic()
    print("Bắt đầu budget validation:", SESSION_BUDGET_MINUTES, "phút; dùng checkpoint đã kiểm tra.")
"""
    return prepare + f"""
if RUN_FULL_VALIDATION:
    if "recovered_paths" not in globals():
        raise RuntimeError("Chạy Cell 10 trước để kiểm tra và cứu checkpoint cho cả hai nhánh.")
    validation_progress = Path(OUTPUT_DIR) / {arm!r} / "progress.pt"
    if str(validation_progress.resolve()) not in recovered_paths:
        print("Chưa có checkpoint nguyên vẹn, đủ batch cho nhánh:", {arm!r}, "; xem output Cell 10.")
    elif remaining_seconds() <= 0:
        print("Hết budget validation; chạy lại Cell 10 rồi Cell 11 để tiếp tục từ case đã lưu.")
    else:
        result = validate_arm(context, OUTPUT_DIR, {arm!r}, max_seconds=remaining_seconds())
        print(result.get("status", "validation_complete"))
else:
    print("Validation đang tắt: đặt RUN_FULL_VALIDATION=True trong Cell 1 để chạy.")
import gc
gc.collect()
torch.cuda.empty_cache()
"""


def runnable_cell_layout(original):
    """Expose one Colab action per numbered cell and a percent-cell export."""
    code=[''.join(cell['source']) for cell in original if cell['cell_type']=='code']
    if len(code)!=9:
        raise ValueError('Unexpected notebook layout; refusing ambiguous cell split')
    mount,gpu=code[1].split('import torch\n',1)
    unpack,dependencies=code[2].split('requirements = ',1)
    stage,verify=code[3].split('expected_hashes = ',1)
    steps=[
        ('Cấu hình đường dẫn và ngân sách',code[0],
         'Sửa đường dẫn dataset, checkpoint epoch 0 và history trên Drive. Giữ RUN_NAME khi chạy tiếp.'),
        ('Mount Google Drive',mount,
         'Chạy và cấp quyền Drive trong cửa sổ Colab.'),
        ('Kiểm tra GPU và file đầu vào','import torch\n'+gpu,
         'Dừng tại đây nếu thiếu GPU hoặc file. Chưa bắt đầu huấn luyện.'),
        ('Giải nén source đã đóng băng',unpack,
         'Cell chứa source nhúng nên được thu gọn; chỉ cần bấm nút chạy. Không cần sửa chuỗi source.'),
        ('Cài dependencies và import worker','requirements = '+dependencies,
         'Giữ Torch/CUDA của runtime, cài phần còn thiếu rồi import source đã nhúng.'),
        ('Copy hoặc giải nén dataset vào runtime',stage,
         'Dữ liệu được stage vào /content để đọc nhanh; bản trên Drive được dùng làm đầu vào.'),
        ('Xác minh dữ liệu và checkpoint','expected_hashes = '+verify,
         'Kiểm tra hash đủ 199 cặp CT/GT, parent epoch 0 và ràng buộc split/group.'),
        ('Train A — alignment hiện tại',code[4],
         'Nhánh tham chiếu: alignment_weight=0.05, lưu tiến độ mỗi 50 batch lên Drive.'),
        ('Train B — chỉ tắt alignment',code[6],
         'Bắt đầu độc lập từ cùng parent; chỉ đặt alignment_weight=0, giữ presence/salience.'),
        ('Validation A — đủ 32 volume',code[5],
         'Chỉ chạy khi A đủ 300 batch; tiếp tục từ case đã lưu nếu phiên trước dừng.'),
        ('Validation B — đủ 32 volume',code[7],
         'Chỉ chạy khi B đủ 300 batch; lưu từng case và bỏ qua các case đã xong.'),
        ('So sánh kết quả hai nhánh',code[8],
         'Chỉ coi comparison là complete khi cả hai nhánh đã validation đủ 32 case.'),
    ]
    intro=''.join(original[0]['source'])
    intro += '\n## Chạy từng cell\n\nChạy lần lượt **Cell 1 → Cell 12** bên dưới. Hai nhánh train trước, rồi mới validation.\n'
    intro += 'Mở file .ipynb bằng Colab/Jupyter để thấy nút chạy từng cell; không dán nội dung JSON vào một code cell.\n'
    result=[]
    def cell(kind,source,hidden=False):
        source=source.strip()+'\n'
        metadata={}
        if hidden:
            metadata={'cellView':'form','jupyter':{'source_hidden':True}}
        value=dict(cell_type=kind,id=hashlib.sha256((kind+source).encode()).hexdigest()[:12],
            metadata=metadata,source=source.splitlines(keepends=True))
        if kind=='code':
            value.update(execution_count=None,outputs=[])
        result.append(value)
    cell('markdown',intro)
    for index,(title,source,description) in enumerate(steps,1):
        cell('markdown',f'## Cell {index} — {title}\n\n{description}')
        cell('code',f'#@title Cell {index} — {title}\n'+source,hidden=index==4)
    cell('markdown',''.join(original[-1]['source'])+
         '\nNếu ngắt phiên: giữ RUN_NAME, chạy lại từ Cell 1. Checkpoint và case validation đã lưu sẽ được dùng lại.\n')
    lines=['# -*- coding: utf-8 -*-',
           '# Percent-cell export: open with a Jupyter-capable editor or copy each # %% block into Colab.']
    for value in result:
        source=''.join(value['source']).rstrip()
        if value['cell_type']=='markdown':
            lines += ['\n# %% [markdown]',*['# '+line for line in source.splitlines()]]
        else:
            lines += ['\n# %%',source]
    CELL_SCRIPT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def build():
    prior=ROOT/'outputs/kaggle_comprehensive_diagnosis'
    script=(prior/'kaggle_suite.py').read_text(encoding='utf-8')
    receipt=json.loads((prior/'package-provenance.json').read_text())
    if hashlib.sha256(script.encode()).hexdigest()!=receipt['script_sha256']:
        raise ValueError('Reviewed source changed')
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(script).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ('BUNDLE','SOURCE_SHA','PILOT_FILES','DATA_HASHES')}
    original=base64.b64decode(constants['BUNDLE'])
    if hashlib.sha256(original).hexdigest()!=constants['SOURCE_SHA']:
        raise ValueError('Frozen archive mismatch')
    with zipfile.ZipFile(io.BytesIO(original)) as source:
        files={name:source.read(name) for name in source.namelist()}
    for name,text in constants['PILOT_FILES'].items():
        if hashlib.sha256(text.encode()).hexdigest()!=receipt['patches'][name]:
            raise ValueError('Helper changed: '+name)
        files[name]=text.encode()
    files['tools/colab_alignment_pair.py']=(ROOT/'tools/colab_alignment_pair.py').read_bytes()
    files['data/colab-content-hashes.json']=(json.dumps(constants['DATA_HASHES'],sort_keys=True,indent=2)+'\n').encode()
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as target:
        for name,content in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            target.writestr(info,content)
    bundle=memory.getvalue()
    bundle_sha=hashlib.sha256(bundle).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'source_snapshot.zip').write_bytes(bundle)
    for name,content in files.items():
        destination=OUT/'test_source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(content)
    cells=[]
    def add(kind,text):
        text=text.strip()+'\n'
        cell=dict(cell_type=kind,id=hashlib.sha256((kind+text).encode()).hexdigest()[:12],
            metadata={},source=text.splitlines(keepends=True))
        if kind=='code':
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown',"""# MERIT — đối chứng alignment trên Colab

Notebook tự chứa source đã đóng băng; **không cần Kaggle API hoặc clone GitHub**.
Hai nhánh bắt đầu từ checkpoint epoch 0, cùng 300 mẫu từ 30 volume, LR 3e-5,
fast attention và frozen BN. Chỉ nhánh thứ hai đặt alignment_weight=0;
presence/salience giữ nguyên. Đây là thử một giả thuyết, chưa phải bản sửa đã xác nhận.

**Chuẩn bị trong Google Drive**, rồi sửa cell cấu hình:

    MyDrive/MERIT_alignment_test/inputs/
      Train/                # toàn bộ các file CT/seg
      epoch0/
        last.pt             # checkpoint epoch 0 gốc, khoảng 2.09 GB
        history.jsonl

MERIT_alignment_test nằm ngay trong My Drive, không nằm trong một thư mục Paper khác.
Notebook dùng đúng /content/drive/MyDrive/MERIT_alignment_test/inputs/Train.
Cell 2 giữ nguyên Drive đã mount; nếu mount trực tiếp Drive của account chủ folder thì không cần shortcut.

Checkpoint/history có sẵn trên máy tại
outputs/kaggle_review_20261116/epoch0_complete/MERIT/fold-00/checkpoints/.
Dataset phải chứa đủ 199 cặp CT/seg đúng dữ liệu cũ; notebook kiểm tra hash giải nén.
Không dùng epoch 1 lỗi hay checkpoint pilot 300 batch làm parent.

Chọn **Runtime → Change runtime type → GPU**, rồi chạy từ trên xuống.
Mặc định ngân sách mỗi phiên là 90 phút; dừng tại ranh giới batch/case và có thể tiếp tục
bằng cùng notebook, đường dẫn và RUN_NAME. Kết quả lưu trên Drive, dữ liệu xử lý ở runtime local.
Dành khoảng 7 GB Drive trống cho hai checkpoint và file ghi tạm.
Không tự mở thêm job hay chạy cả bộ 13 nhánh cũ.
""")
    add('code',"""# Chỉ sửa các đường dẫn và ngân sách tại đây.
DRIVE_ROOT = "/content/drive/MyDrive/MERIT_alignment_test"
SHARED_FOLDER_URL = "https://drive.google.com/drive/folders/1wW7uej60ZBLOL0_WVv-uBDQsLPkr6ar8"
SHARED_PROJECT_PATH = ""  # nếu shortcut có tên/vị trí khác, điền đường dẫn mount vào đây
DATASET_SOURCE = DRIVE_ROOT + "/inputs/Train"
EPOCH0_CHECKPOINT = DRIVE_ROOT + "/inputs/epoch0/last.pt"
EPOCH0_HISTORY = DRIVE_ROOT + "/inputs/epoch0/history.jsonl"
FORCE_REMOUNT_DRIVE = False  # giữ Drive đang mount; chỉ bật True khi muốn mount lại
RUN_NAME = "alignment_pair_300_v1"
TOTAL_BATCHES = 300
CHECKPOINT_EVERY = 50
SESSION_BUDGET_MINUTES = 90
RUN_FULL_VALIDATION = True
PROJECT_DIR = "/content/merit_alignment_project"
SCRATCH_DIR = "/content/merit_alignment_scratch"
DATA_CACHE_DIR = "/content/merit_alignment_dataset"
""")
    add('code',drive_mount_cell_source()+"""
import torch
def resolve_notebook_inputs(inputs_dir, dataset_source, checkpoint, history):
    inputs_dir = Path(inputs_dir).expanduser()
    if not inputs_dir.is_dir():
        project_root = inputs_dir.parent
        my_drive = project_root.parent
        actual = [p.name for p in my_drive.iterdir()] if my_drive.is_dir() else []
        project_items = [p.name for p in project_root.iterdir()] if project_root.is_dir() else []
        raise FileNotFoundError("Không thấy đúng đường dẫn: " + str(inputs_dir)
            + ". MERIT_alignment_test phải ở ngay My Drive của account vừa mount."
            + " Các mục hiện có trong My Drive: " + repr(actual[:25])
            + "; các mục trong MERIT_alignment_test: " + repr(project_items[:25])
            + ". Chạy lại Cell 2 và chọn đúng Google account; kiểm tra chính tả/tên thư mục.")
    dataset = Path(dataset_source).expanduser()
    if not dataset.exists():
        # Folder names do not have to be Train; accept any nested NIfTI layout.
        nifti = next((p for p in inputs_dir.rglob("*")
                      if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))), None)
        if nifti is not None:
            dataset = inputs_dir
        else:
            archives = sorted(p for p in inputs_dir.rglob("*.zip") if p.is_file())
            if len(archives) == 1:
                dataset = archives[0]
            else:
                raise FileNotFoundError("Chưa thấy CT/seg dưới " + str(inputs_dir)
                    + ". Đợi upload xong hoặc sửa DATASET_SOURCE. ZIP tìm thấy: " + repr(archives))
    resolved = []
    issues = []
    for configured, filename in ((checkpoint, "last.pt"), (history, "history.jsonl")):
        path = Path(configured).expanduser()
        if path.is_file():
            resolved.append(str(path))
            continue
        found = sorted(p for p in inputs_dir.rglob(filename) if p.is_file())
        if len(found) == 1:
            resolved.append(str(found[0]))
        elif not found:
            issues.append("Không tìm thấy " + filename + ". Đường dẫn đã cấu hình: " + str(path)
                + "; đã tìm thêm bên dưới " + str(inputs_dir)
                + ". Upload file checkpoint epoch 0 vào inputs/epoch0, hoặc sửa đường dẫn trong Cell 1.")
        else:
            issues.append("Có nhiều " + filename + ": đặt đường dẫn rõ trong Cell 1: "
                + repr([str(p) for p in found]))
    if issues:
        epoch0_dir = inputs_dir / "epoch0"
        epoch0_items = sorted(p.name for p in epoch0_dir.iterdir()) if epoch0_dir.is_dir() else []
        raise FileNotFoundError("Đã tìm thấy đường dẫn data: " + str(dataset)
            + "\\nCác mục trong inputs/epoch0: " + repr(epoch0_items[:20])
            + "\\n" + "\\n".join(issues))
    return str(dataset), resolved[0], resolved[1]

DATASET_SOURCE, EPOCH0_CHECKPOINT, EPOCH0_HISTORY = resolve_notebook_inputs(
    Path(DRIVE_ROOT) / "inputs", DATASET_SOURCE, EPOCH0_CHECKPOINT, EPOCH0_HISTORY)
print("Dataset:", DATASET_SOURCE)
print("Epoch 0:", EPOCH0_CHECKPOINT)
print("History:", EPOCH0_HISTORY)
dataset_folder = Path(DATASET_SOURCE)
if dataset_folder.is_dir():
    nifti_files = [p for p in dataset_folder.rglob("*")
                   if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))]
    if not nifti_files:
        raise FileNotFoundError("Tìm thấy thư mục data nhưng chưa có file NIfTI: "
            + str(dataset_folder) + ". Hãy đợi upload xong các file CT/seg.")
    print("Số file NIfTI tìm thấy:", len(nifti_files), "(đủ bộ dữ liệu thường là 398 file)")
if not torch.cuda.is_available():
    raise RuntimeError("Chọn GPU runtime trước khi chạy. Notebook không tự train bằng CPU.")
if torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3:
    raise RuntimeError("Phép thử native 512 cần GPU khoảng 16 GB VRAM trở lên; không tự giảm độ phân giải.")
if not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_NAME):
    raise ValueError("RUN_NAME chỉ gồm chữ, số, _ hoặc -")
if TOTAL_BATCHES != 300 or SESSION_BUDGET_MINUTES <= 0:
    raise ValueError("Giữ budget 300 batch và ngân sách phiên dương")
SESSION_STARTED = time.monotonic()
def remaining_seconds():
    return max(0.0, SESSION_BUDGET_MINUTES * 60 - (time.monotonic() - SESSION_STARTED) - 120)
OUTPUT_DIR = Path(DRIVE_ROOT) / "runs" / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("GPU:", torch.cuda.get_device_name(0), "Torch:", torch.__version__)
print("Kết quả:", OUTPUT_DIR)
print("Ngân sách phiên:", SESSION_BUDGET_MINUTES, "phút; chừa 2 phút ghi dữ liệu")
""")
    add('markdown',"""## Nạp source và dependencies

Giữ Torch/CUDA sẵn có của runtime; cài phần còn thiếu theo phiên bản đã dùng.
Môi trường GPU/Torch phải khớp khi tiếp tục một cặp thử nghiệm.
Chuyển Kaggle → Colab không được coi là tái lập bit-for-bit môi trường lịch sử;
hai nhánh mới được so trong cùng môi trường Colab.
""")
    add('code',"SOURCE_BUNDLE_B64 = "+repr(base64.b64encode(bundle).decode())+
        '\nSOURCE_BUNDLE_SHA256 = '+repr(bundle_sha)+"""
import base64, hashlib, io, json, sys, zipfile, subprocess
from importlib import metadata
from packaging.specifiers import SpecifierSet
payload = base64.b64decode(SOURCE_BUNDLE_B64)
if hashlib.sha256(payload).hexdigest() != SOURCE_BUNDLE_SHA256:
    raise ValueError("Source snapshot hash mismatch")
project = Path(PROJECT_DIR).resolve()
project.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        if not (project / member.filename).resolve().is_relative_to(project):
            raise ValueError("Invalid source archive path")
    archive.extractall(project)
requirements = {"numpy": ">=1.24", "scipy": ">=1.10", "nibabel": ">=5",
                "PyYAML": ">=6", "monai": "==1.5.2", "einops": "==0.8.2"}
if metadata.version("torch") not in SpecifierSet(">=2.4.1"):
    raise RuntimeError("Torch runtime quá cũ; chọn runtime Colab mới, không tự thay Torch/CUDA")
needed = []
for package, spec in requirements.items():
    try:
        okay = metadata.version(package) in SpecifierSet(spec)
    except metadata.PackageNotFoundError:
        okay = False
    if not okay:
        needed.append(package + spec)
if needed:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(project)], check=True)
os.chdir(project)
sys.path[:0] = [str(project / "src"), str(project)]
import merit_covid
if Path(merit_covid.__file__).resolve() != project / "src/merit_covid/__init__.py":
    raise RuntimeError("Python đang giữ package từ lần khác; restart runtime rồi chạy lại")
import importlib
importlib.invalidate_caches()
alignment_worker = importlib.import_module("tools.colab_alignment_pair")
alignment_worker = importlib.reload(alignment_worker)
stage_dataset = alignment_worker.stage_dataset
prepare = alignment_worker.prepare
run_arm = alignment_worker.run_arm
validate_arm = alignment_worker.validate_arm
compare = alignment_worker.compare
write_json = alignment_worker.write_json
print("Source SHA256:", SOURCE_BUNDLE_SHA256)
""")
    add('markdown',"""## Chuẩn bị và xác minh dữ liệu/checkpoint

Copy/giải nén vào runtime local để tránh đọc hàng nghìn slab trực tiếp từ Drive.
Chỉ đổi đường dẫn metadata sau khi tất cả CT/mask khớp hash; giữ nguyên split,
group, intensities, kích thước native, trọng số và optimizer moments.
Checkpoint Kaggle lưu RNG hai GPU; Colab một GPU dùng đúng stream GPU 0.
""")
    add('code',"""dataset_local = stage_dataset(DATASET_SOURCE, DATA_CACHE_DIR)
expected_hashes = json.loads((project / "data/colab-content-hashes.json").read_text())
context = prepare(project, dataset_local, EPOCH0_CHECKPOINT, EPOCH0_HISTORY, expected_hashes,
    SCRATCH_DIR,
    parent_sha256="22596971dc0f3ff7b6796756a61cf61187ded97472a9bdf6d754766165cd6934",
    source_sha256=SOURCE_BUNDLE_SHA256)
write_json(OUTPUT_DIR / "input-provenance.json", context)
PROBE_IDS = ["volume-covid19-A-0014", "volume-covid19-A-0319", "volume-covid19-A-0347"]
print("Input đã xác minh; phiên còn khoảng", round(remaining_seconds()/60, 1), "phút")
""")
    add('markdown',"""## Chạy hai nhánh

Mỗi nhánh lưu một checkpoint tiến độ lên Drive sau mỗi 50 batch.
Resume khôi phục model/optimizer/scheduler/scaler/RNG và bỏ qua batch đã xong.
Ngắt runtime đột ngột có thể mất phần sau checkpoint gần nhất.
Validation lưu từng case; chỉ tính điểm toàn tập khi đủ 32 case.
Giữ nguyên RUN_NAME khi tiếp tục cùng phép thử; chỉ đổi khi bắt đầu cặp mới.
""")
    for arm,label in [('reference_alignment','A — alignment hiện tại'),('alignment_off','B — chỉ tắt alignment')]:
        add('markdown','## '+label)
        add('code',f"""if remaining_seconds() <= 0:
    print("Hết ngân sách phiên; chạy tiếp trong phiên mới với cùng RUN_NAME.")
else:
    report = run_arm(context, OUTPUT_DIR, {arm!r}, total_batches=TOTAL_BATCHES,
        checkpoint_every=CHECKPOINT_EVERY, probe_every=50,
        max_seconds=remaining_seconds(), probe_ids=PROBE_IDS)
    print(report["arm"], report["status"], report["processed_batches"], "/", TOTAL_BATCHES)
import gc
gc.collect()
torch.cuda.empty_cache()
""")
        add('code',validation_cell_source(arm))
    add('markdown',"""## Kết quả và tiếp tục

Chỉ so bảng cuối khi cả hai nhánh đủ 32 case. Ba lát probe không thay điểm native toàn volume.
Không chỉnh threshold theo điểm đẹp nhất. Khi reconnect, mount lại Drive, giữ nguyên cấu hình/RUN_NAME
và chạy lại từ trên: không train lại nhánh đã đủ batch, không dự đoán lại case đã lưu.
Nếu GPU/Torch thay đổi, cần môi trường cũ hoặc RUN_NAME mới để tránh ghép hai điều kiện khác nhau.

Drive giữ progress.pt, arm-report.json, validation-partial.json, validation-native.json cho từng nhánh,
và comparison.json cho cặp thử nghiệm. progress.pt là checkpoint riêng của phép thử, không phải epoch hoàn tất.
""")
    add('code',"""comparison = compare(OUTPUT_DIR)
print("Comparison status:", comparison["status"])
if comparison["status"] == "partial":
    print("Chưa đủ cả hai nhánh: xem tiến độ trên Drive rồi chạy tiếp khi có runtime.")
else:
    print("Hai nhánh khớp restart/mẫu. Xem Dice, precision, recall, FP và dự đoán rỗng.")
print("Kết quả:", OUTPUT_DIR)
""")
    add('markdown',"""Notebook được kiểm tra schema, source và resume bằng CPU cục bộ trước khi giao;
chưa được thực thi trong tài khoản Colab của bạn.
Hướng dẫn mount Drive: [Colab I/O chính thức](https://colab.research.google.com/notebooks/io.ipynb).
Dữ liệu cần giữ được ghi ra Drive; không dựa vào việc /content còn tồn tại ở phiên sau.
""")
    cells=runnable_cell_layout(cells)
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),accelerator='GPU',
        colab=dict(name=NOTEBOOK.name,provenance=[],gpuType='T4')))
    NOTEBOOK.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    manifest=dict(notebook=str(NOTEBOOK.relative_to(ROOT)),
        notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        source_bundle_sha256=bundle_sha,upstream_frozen_source_sha256=constants['SOURCE_SHA'],
        parent_checkpoint_sha256='22596971dc0f3ff7b6796756a61cf61187ded97472a9bdf6d754766165cd6934',
        files={name:hashlib.sha256(content).hexdigest() for name,content in files.items()},
        arms=['reference_alignment','alignment_off'],total_batches_per_arm=300,
        validation_cases=32,hosted_notebook_executed=False,numbered_code_cells=12,
        percent_cell_export=str(CELL_SCRIPT.relative_to(ROOT)),
        percent_cell_export_sha256=hashlib.sha256(CELL_SCRIPT.read_bytes()).hexdigest())
    (OUT/'notebook-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(NOTEBOOK)
    print('CELLS',len(cells),'BUNDLE_BYTES',len(bundle))
    return notebook


if __name__=='__main__':
    build()
