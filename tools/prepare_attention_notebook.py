"""Build a self-contained Colab notebook from the current local source and CT identities."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import textwrap
import zipfile

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ("volume-covid19-A-0025", "volume-covid19-A-0295", "volume-covid19-A-0387")
DATASET_REVISION = "e6ce5e1c01f59b29843e6a969655a42d33a52729"


def build_payload(root=ROOT):
    split_bytes = (root / "splits/covid19_20_5fold.json").read_bytes().replace(b"\r\n", b"\n")
    split = json.loads(split_bytes)
    train = next(row for row in split["folds"] if row["fold"] == 0)["train"]
    specs = []
    for case_id in CASE_IDS:
        if case_id not in train["case_ids"] or case_id not in train["group_ids"]:
            raise ValueError(f"{case_id} must be TRAIN fold 0")
        ct, gt = (root / "data/covid19_20/Train" / f"{case_id}_{kind}.nii.gz" for kind in ("ct", "seg"))
        mask = nib.load(gt).get_fdata(dtype=np.float32)
        positives = np.flatnonzero((mask > 0).any(axis=(0, 1)))
        if not len(positives) or (mask[:, :, :6] > 0).any():
            raise ValueError("Pilot needs a positive master and an empty boundary slab")
        specs.append(dict(case_id=case_id, group_id=case_id, fold=0,
            ct_name=ct.name, gt_name=gt.name, shape_xyz=list(mask.shape),
            masters=[int(positives[len(positives) // 2]), 0],
            sha256={"ct": hashlib.sha256(ct.read_bytes()).hexdigest(),
                    "gt": hashlib.sha256(gt.read_bytes()).hexdigest()},
            file_sizes={"ct": ct.stat().st_size, "gt": gt.stat().st_size},
            split_sha256_lf=hashlib.sha256(split_bytes).hexdigest(),
            manifest_fingerprint=split["manifest_fingerprint"]))
    paths = []
    for directory in ("src", "configs", "splits"):
        paths.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    paths.extend(root / name for name in (
        "pyproject.toml", "data/covid19_20/cases.json", "tools/merit_attention_pilot.py",
        "tools/merit_real_case_trial.py", "tools/merit_checkpoint_trial.py",
        "tests/test_mer_attention_execution.py", "tests/test_production_checkpointing.py", "tests/conftest.py"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(2026, 11, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue(), specs


def cell(kind, source):
    result = dict(cell_type=kind, id=hashlib.sha256(source.encode()).hexdigest()[:12],
                  metadata={}, source=textwrap.dedent(source).strip().splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def runtime_profile_cell():
    profiler_source = (ROOT / "tools/merit_runtime_profile.py").read_text(encoding="utf-8")
    notebook_path = ROOT / "notebooks/colab_mer_attention_pilot.ipynb"
    if notebook_path.is_file():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        bootstrap = next("".join(c["source"]) for c in notebook["cells"]
                         if c["cell_type"] == "code" and "SOURCE_BUNDLE_B64 = " in "".join(c["source"]))
    else:
        bootstrap = ""
    source = ("PROFILE_SOURCE = " + repr(profiler_source) + "\nPROFILE_SHA256 = "
              + repr(hashlib.sha256(profiler_source.encode()).hexdigest()) + "\nPROFILE_BOOTSTRAP = " + repr(bootstrap) + "\n")
    source += textwrap.dedent('''
        # Chạy riêng cell này: tự khởi tạo nếu kernel mới và dùng lại dữ liệu cache.
        # Một ca A-0025 đã cache. Mỗi backend: 1 warmup + 2 step đo + 1 step profiler.
        # Không chạy lại toàn bộ pilot. Đây là chẩn đoán, chưa đổi production config.
        if not all(name in globals() for name in ("PROJECT", "DATA", "RUN", "ENV", "BASE")):
            if not PROFILE_BOOTSTRAP:
                raise RuntimeError("Không tìm thấy snapshot source. Chạy cell đầu tiên của notebook.")
            print("Khởi tạo lại source cho kernel mới; không chạy lại pilot trước.")
            exec(PROFILE_BOOTSTRAP, globals())
        PROFILE_RUN = BASE / ("runtime_profile_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        PROFILE_RUN.mkdir()
        assert hashlib.sha256(PROFILE_SOURCE.encode()).hexdigest() == PROFILE_SHA256
        worker = PROFILE_RUN / "merit_runtime_profile.py"
        worker.write_text(PROFILE_SOURCE)
        (PROFILE_RUN / "provenance.json").write_text(json.dumps(dict(source_archive_sha256=SOURCE_SHA256,
            profile_code_sha256=PROFILE_SHA256, scope="one cached TRAIN case; runtime diagnosis"), indent=2))
        import shutil
        for name in ("summary.json", "environment.json", "legacy.json", "optimized.json"):
            prior = RUN / name
            if prior.is_file():
                shutil.copy2(prior, PROFILE_RUN / ("previous-" + name))
        profiles = {}
        for mode in ("auto", "efficient"):
            destination = PROFILE_RUN / (mode + ".json")
            command = [sys.executable, "-u", str(worker), "--config", str(PROJECT / "configs/merit_native512.yaml"),
                "--data-dir", str(DATA), "--specs", str(RUN / "case-specifications.json"),
                "--output", str(destination), "--backend", mode]
            with (PROFILE_RUN / (mode + ".log")).open("w") as log:
                process = subprocess.Popen(command, cwd=PROJECT, env=ENV, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end="", flush=True)
                code = process.wait()
            result = json.loads(destination.read_text()) if destination.is_file() else dict(status="FAIL", error="Worker did not write a report")
            if code:
                result["status"] = "FAIL"
            result["exit_code"] = code
            destination.write_text(json.dumps(result, indent=2))
            profiles[mode] = result
            print("RUNTIME DETAILS", mode)
            print(json.dumps({key: result.get(key) for key in ("status", "error", "two_step_mean_seconds",
                "mer_first_call", "attention_ops", "stages", "top_self_cpu", "top_self_device", "amp_fp32_retries")}, indent=2))
        same = (profiles["auto"].get("initial_state_sha256") is not None and
                profiles["auto"].get("initial_state_sha256") == profiles["efficient"].get("initial_state_sha256"))
        diagnosis = dict(status="PASS" if same and all(p.get("status") == "PASS" for p in profiles.values()) else "FAIL",
            same_initial_weights=same, source_archive_sha256=SOURCE_SHA256, profile_code_sha256=PROFILE_SHA256,
            measured_steps_per_mode=2, note="Profiler step and warmup excluded; two-step timing is a small diagnostic, not established throughput.")
        if diagnosis["status"] == "PASS":
            diagnosis["observed_auto_over_efficient_ratio"] = profiles["auto"]["two_step_mean_seconds"] / profiles["efficient"]["two_step_mean_seconds"]
        (PROFILE_RUN / "profile-summary.json").write_text(json.dumps(diagnosis, indent=2))
        profile_zip = PROFILE_RUN.with_suffix(".zip")
        with zipfile.ZipFile(profile_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(PROFILE_RUN.iterdir()):
                if path.is_file() and path.suffix in (".json", ".log"):
                    archive.write(path, path.name)
        print(json.dumps(diagnosis, indent=2))
        print("GỬI LẠI ZIP CHẨN ĐOÁN:", profile_zip)
        try:
            from google.colab import files
            files.download(str(profile_zip))
        except Exception:
            print("Trong VS Code: Download file ZIP trên bằng Colab Explorer.")
    ''')
    entry = cell("code", source)
    entry["metadata"]["tags"] = ["merit-runtime-profile"]
    return entry


def factorized_pilot_cell():
    paths = ("tools/merit_factorized_attention.py", "tools/merit_factorized_pilot.py",
             "tests/test_mer_factorized_attention.py")
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in paths}
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    path = ROOT / "notebooks/colab_mer_attention_pilot.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    bootstrap = next("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == "code" and "SOURCE_BUNDLE_B64 = " in "".join(c["source"]))
    source = "FACTOR_SOURCES = " + repr(sources) + "\nFACTOR_HASHES = " + repr(hashes) + "\nFACTOR_BOOTSTRAP = " + repr(bootstrap) + "\n"
    source += textwrap.dedent('''
        # Chạy riêng cell này trên T4. Hai lượt dùng ba ca train đã cache.
        # Giữ full context MER; biến đổi bias XY bằng đại số, không resize.
        if not all(name in globals() for name in ("PROJECT", "DATA", "RUN", "ENV", "BASE")):
            print("Khởi tạo source cho kernel mới.")
            exec(FACTOR_BOOTSTRAP, globals())
        import shutil
        FACTOR_RUN = BASE / ("factorized_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        FACTOR_RUN.mkdir()
        CANDIDATE_PROJECT = FACTOR_RUN / "source"
        shutil.copytree(PROJECT, CANDIDATE_PROJECT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name, source_text in FACTOR_SOURCES.items():
            if hashlib.sha256(source_text.encode()).hexdigest() != FACTOR_HASHES[name]:
                raise ValueError("Factorized source hash mismatch")
            path = CANDIDATE_PROJECT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source_text, encoding="utf-8")
        FACTOR_ENV = dict(ENV, PYTHONPATH=os.pathsep.join([str(CANDIDATE_PROJECT / "src"), str(CANDIDATE_PROJECT)]))
        (FACTOR_RUN / "source-provenance.json").write_text(json.dumps(dict(base_source_sha256=SOURCE_SHA256,
            candidate_source_sha256=FACTOR_HASHES), indent=2))
        # Kiểm tra 8 CPU và 4 CUDA: output, input/parameter gradients,
        # checkpoint recomputation, physical positions và context mask.
        checks = subprocess.run([sys.executable, "-m", "pytest", "tests/test_mer_factorized_attention.py",
            "-q", "--tb=short", "-p", "no:cacheprovider"], cwd=CANDIDATE_PROJECT,
            env=FACTOR_ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
        (FACTOR_RUN / "equivalence.log").write_text(checks.stdout, encoding="utf-8")
        print(checks.stdout)
        import re
        match = re.search(r"(\\d+) passed", checks.stdout)
        regression_pass = checks.returncode == 0 and match is not None and int(match[1]) == 12
        reports = {}
        if regression_pass:
            for label, script in (("reference", "tools/merit_attention_pilot.py"),
                                  ("factorized", "tools/merit_factorized_pilot.py")):
                report_path = FACTOR_RUN / (label + ".json")
                command = [sys.executable, "-u", script, "--config", str(CANDIDATE_PROJECT / "configs/merit_native512.yaml"),
                    "--data-dir", str(DATA), "--specs", str(RUN / "case-specifications.json"),
                    "--output", str(report_path), "--rounds", "1"]
                if label == "reference":
                    command += ["--variant", "optimized"]
                with (FACTOR_RUN / (label + ".log")).open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=CANDIDATE_PROJECT, env=FACTOR_ENV, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
                    for line in process.stdout:
                        log.write(line); log.flush(); print(line, end="", flush=True)
                    exit_code = process.wait()
                result = json.loads(report_path.read_text()) if report_path.is_file() else dict(status="FAIL", error="No worker report")
                if exit_code:
                    result["status"] = "FAIL"
                result["exit_code"] = exit_code
                report_path.write_text(json.dumps(result, indent=2))
                reports[label] = result
        same = (len(reports) == 2 and reports["reference"].get("initial_state_sha256") is not None
                and reports["reference"].get("initial_state_sha256") == reports["factorized"].get("initial_state_sha256"))
        summary = dict(status="PASS" if regression_pass and same and all(r.get("status") == "PASS" for r in reports.values()) else "FAIL",
            regression_pass=regression_pass, same_initial_weights=same, cases=[s["case_id"] for s in CASE_SPECS],
            base_source_sha256=SOURCE_SHA256, candidate_source_sha256=FACTOR_HASHES,
            scope="3 TRAIN cases, 6 measured updates each after warmup; no validation or accuracy claim")
        if summary["status"] == "PASS":
            old, new = reports["reference"], reports["factorized"]
            summary.update(reference_mean_step_seconds=old["mean_step_seconds"], factorized_mean_step_seconds=new["mean_step_seconds"],
                observed_speed_ratio=old["mean_step_seconds"] / new["mean_step_seconds"],
                reference_peak_allocated_mib=old["peak_allocated_mib"], factorized_peak_allocated_mib=new["peak_allocated_mib"],
                reference_fp32_retries=old["amp_fp32_retries"], factorized_fp32_retries=new["amp_fp32_retries"])
        (FACTOR_RUN / "summary.json").write_text(json.dumps(summary, indent=2))
        candidate_zip = FACTOR_RUN.with_suffix(".zip")
        with zipfile.ZipFile(candidate_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in FACTOR_RUN.iterdir():
                if path.is_file() and path.suffix in (".json", ".log"):
                    archive.write(path, path.name)
        print(json.dumps(summary, indent=2))
        print("KẾT QUẢ TEST MER:", candidate_zip)
        try:
            from google.colab import files
            files.download(str(candidate_zip))
        except Exception:
            print("Trong VS Code: Download ZIP bằng Colab Explorer.")
    ''')
    result = cell("code", source)
    result["metadata"]["tags"] = ["merit-factorized-pilot"]
    return result


def make_notebook(bundle, specs, commit):
    sha = hashlib.sha256(bundle).hexdigest()
    setup = f'''SOURCE_BUNDLE_B64 = {base64.b64encode(bundle).decode()!r}
SOURCE_SHA256 = {sha!r}
SOURCE_COMMIT = {commit!r}
CASE_SPECS = {specs!r}
DATASET_REVISION = {DATASET_REVISION!r}
MEASURED_ROUNDS = 2  # 12 measured updates + 1 warmup per variant; max 3 rounds.
'''
    setup += textwrap.dedent('''
        import base64, datetime, hashlib, io, json, os, subprocess, sys, uuid, zipfile
        from pathlib import Path
        BASE = Path(os.environ.get("MERIT_ATTENTION_HOME", "/content/merit-attention-pilot"))
        BASE.mkdir(parents=True, exist_ok=True)
        RUN = BASE / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        RUN.mkdir()
        PROJECT = BASE / ("source_" + SOURCE_SHA256[:16])
        DATA = Path(os.environ.get("MERIT_ATTENTION_DATA", str(BASE / "data")))
        DATA.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(SOURCE_BUNDLE_B64)
        assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for name in archive.namelist():
                target = (PROJECT / name).resolve()
                if not target.is_relative_to(PROJECT.resolve()):
                    raise ValueError("Invalid archive path")
            archive.extractall(PROJECT)
            SOURCE_FILES = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
        (RUN / "case-specifications.json").write_text(json.dumps(CASE_SPECS, indent=2))
        (RUN / "source-provenance.json").write_text(json.dumps(dict(source_archive_sha256=SOURCE_SHA256,
            git_commit=SOURCE_COMMIT, includes_local_changes=True, source_files=SOURCE_FILES), indent=2))
        ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PROJECT / "src"), str(PROJECT)]),
            CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        PREPARATION_ERROR = None
        print("Source:", SOURCE_SHA256)
        print("Cases:", [s["case_id"] for s in CASE_SPECS])
        print("Results:", RUN)
    ''')
    cells = [cell("markdown", '''
        # MERIT: test attention cũ–mới trên 3 ca CT/GT thật

        Bản sửa giữ backend attention nhất quán qua forward và backward trong bài test
        activation checkpointing. Dữ liệu đã tải từ lần trước được dùng lại sau kiểm tra hash.

        Chọn **kernel Colab GPU** rồi **Run All**. Notebook có sẵn snapshot source mới;
        không cần clone GitHub hoặc tải code lên. Chỉ tải **6 file (~48 MB)** của
        **A-0025, A-0295, A-0387**, so hash với dữ liệu local của bạn và xác nhận TRAIN fold 0.

        Test giữ nguyên input **[1,1,11,512,512]**, native intensity, TEP/MER/PCG,
        loss và activation checkpointing. Mỗi ca có một master positive và một master empty.
        Có 9 kiểm tra CUDA nhỏ trước khi đo hai process riêng: attention cũ và mới.
        Mỗi process chạy **1 warmup + 12 update được đo**, không train cả epoch.

        Cuối notebook tải ZIP báo cáo/log. Trong VS Code, nếu download tự động không
        hoạt động, tải ZIP từ thư mục `/content/merit-attention-pilot` bằng Colab Explorer.
        ZIP không chứa CT/GT hoặc weights. Cần khoảng **3 GB disk trống** cho checkpoint
        chẩn đoán tạm; file này được xóa sau kiểm tra. Đây chưa phải kết quả accuracy.
    '''), cell("code", setup), cell("code", '''
        # Cài phần còn thiếu; giữ nguyên Torch/CUDA của runtime Colab.
        import importlib.metadata as metadata
        from packaging.specifiers import SpecifierSet
        requirements = {"numpy": ">=1.24", "scipy": ">=1.10", "nibabel": ">=5",
                        "PyYAML": ">=6", "monai": "==1.5.2", "einops": "==0.8.2", "pytest": ">=7"}
        try:
            if metadata.version("torch") not in SpecifierSet(">=2.4.1"):
                raise RuntimeError("Runtime cần Torch >=2.4.1; notebook không thay Torch/CUDA.")
            missing = []
            for package, version in requirements.items():
                try:
                    installed = metadata.version(package)
                except metadata.PackageNotFoundError:
                    installed = None
                if installed is None or installed not in SpecifierSet(version):
                    missing.append(package + version)
            if missing:
                result = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", *missing],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=ENV)
                (RUN / "installation.log").write_text(result.stdout)
                if result.returncode:
                    raise RuntimeError("Cài dependency thất bại; xem installation.log")
            probe = subprocess.run([sys.executable, "-c", "import torch,json; print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, available=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)))"],
                capture_output=True, text=True, env=ENV)
            (RUN / "environment-probe.log").write_text(probe.stdout + probe.stderr)
            if probe.returncode:
                raise RuntimeError("Không khởi tạo được Torch/CUDA; xem environment-probe.log")
            ENVIRONMENT = json.loads(probe.stdout.strip().splitlines()[-1])
            if not ENVIRONMENT["available"]:
                raise RuntimeError("Chưa có GPU. Chọn kernel/runtime Colab GPU rồi Run All lại.")
            (RUN / "environment.json").write_text(json.dumps(ENVIRONMENT, indent=2))
            print(ENVIRONMENT)
        except Exception as error:
            PREPARATION_ERROR = str(error)
            print("PREPARATION FAILED:", PREPARATION_ERROR)
    '''), cell("code", '''
        # Chỉ tải 3 ca; file cache chỉ dùng lại nếu đúng SHA-256.
        import time, urllib.request
        def digest(path):
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            return value.hexdigest()
        if PREPARATION_ERROR is None:
            try:
                for spec in CASE_SPECS:
                    for kind, name in (("ct", spec["ct_name"]), ("gt", spec["gt_name"])):
                        target = DATA / name
                        if not target.is_file() or digest(target) != spec["sha256"][kind]:
                            url = f"https://huggingface.co/datasets/MedOtter/COVID-19-20/resolve/{DATASET_REVISION}/Train/{name}"
                            partial = target.with_suffix(target.suffix + ".partial")
                            for attempt in range(3):
                                try:
                                    with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as sink:
                                        for block in iter(lambda: source.read(1024 * 1024), b""):
                                            sink.write(block)
                                    if digest(partial) != spec["sha256"][kind]:
                                        raise ValueError("Downloaded data differs from your local CT/GT: " + name)
                                    partial.replace(target)
                                    break
                                except Exception:
                                    if attempt == 2:
                                        raise
                                    time.sleep(2 * (attempt + 1))
                        print("Hash OK:", name)
                (RUN / "preparation.json").write_text(json.dumps(dict(status="PASS", cases=len(CASE_SPECS), files=6)))
            except Exception as error:
                PREPARATION_ERROR = str(error)
        if PREPARATION_ERROR:
            (RUN / "preparation.json").write_text(json.dumps(dict(status="FAIL", error=PREPARATION_ERROR)))
            print("PREPARATION FAILED:", PREPARATION_ERROR)
    '''), cell("code", '''
        # 8 MER CUDA cases + 1 compact full-model CUDA/AMP checkpoint comparison.
        import re
        REGRESSION = dict(status="NOT_RUN")
        if PREPARATION_ERROR is None:
            try:
                result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_mer_attention_execution.py",
                    "tests/test_production_checkpointing.py", "-k", "efficient_backend or production_cuda_amp",
                    "-q", "--tb=short", "-p", "no:cacheprovider"], cwd=PROJECT, env=ENV, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
                (RUN / "cuda-regression.log").write_text(result.stdout)
                passed = re.search(r"(\\d+) passed", result.stdout)
                REGRESSION = dict(status="PASS" if result.returncode == 0 and passed and int(passed[1]) == 9 else "FAIL",
                    exit_code=result.returncode, passed=int(passed[1]) if passed else 0)
                print(result.stdout)
            except Exception as error:
                REGRESSION = dict(status="FAIL", error=str(error))
        (RUN / "cuda-regression.json").write_text(json.dumps(REGRESSION, indent=2))
        print("CUDA regression:", REGRESSION)
    '''), cell("code", '''
        # Hai process độc lập; cùng init, dữ liệu, thứ tự, optimizer và warmup.
        # Hiển thị tiến độ trực tiếp và giữ cả stderr/stdout trong log.
        REPORTS = {}
        if PREPARATION_ERROR is None and REGRESSION["status"] == "PASS":
            for variant in ("legacy", "optimized"):
                report_path = RUN / (variant + ".json")
                command = [sys.executable, "-u", "tools/merit_attention_pilot.py", "--config",
                    str(PROJECT / "configs/merit_native512.yaml"), "--data-dir", str(DATA),
                    "--specs", str(RUN / "case-specifications.json"), "--output", str(report_path),
                    "--variant", variant, "--rounds", str(MEASURED_ROUNDS)]
                with (RUN / (variant + ".log")).open("w") as log:
                    process = subprocess.Popen(command, cwd=PROJECT, env=ENV, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    exit_code = process.wait()
                report = json.loads(report_path.read_text()) if report_path.is_file() else dict(status="FAIL", error="Worker exited without report")
                report.update(worker_exit_code=exit_code, source_archive_sha256=SOURCE_SHA256)
                if exit_code:
                    report["status"] = "FAIL"
                report_path.write_text(json.dumps(report, indent=2))
                REPORTS[variant] = report
        else:
            print("Bỏ qua benchmark vì chuẩn bị/CUDA regression chưa PASS. ZIP vẫn được tạo ở cell cuối.")
    '''), cell("code", '''
        # Luôn tạo ZIP báo cáo, cả khi download, test hoặc worker thất bại.
        source_unchanged = all((PROJECT / name).is_file() and
            hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == expected for name, expected in SOURCE_FILES.items())
        same_initial = (len(REPORTS) == 2 and REPORTS["legacy"].get("initial_state_sha256") is not None
            and REPORTS["legacy"].get("initial_state_sha256") == REPORTS["optimized"].get("initial_state_sha256"))
        passed = (PREPARATION_ERROR is None and REGRESSION["status"] == "PASS" and same_initial
            and source_unchanged and all(r.get("status") == "PASS" for r in REPORTS.values()))
        summary = dict(status="PASS" if passed else "FAIL", source_archive_sha256=SOURCE_SHA256,
            source_files_unchanged=source_unchanged, same_initial_weights=same_initial,
            preparation_error=PREPARATION_ERROR, cuda_regression=REGRESSION,
            cases=[s["case_id"] for s in CASE_SPECS], measured_rounds=MEASURED_ROUNDS,
            full_training=False, accuracy_evaluated=False,
            next_step="Review this pilot before preparing a new Kaggle run.")
        if passed:
            old, new = REPORTS["legacy"], REPORTS["optimized"]
            summary.update(legacy_mean_step_seconds=old["mean_step_seconds"],
                optimized_mean_step_seconds=new["mean_step_seconds"],
                observed_step_speed_ratio=old["mean_step_seconds"] / new["mean_step_seconds"],
                legacy_peak_allocated_mib=old["peak_allocated_mib"],
                optimized_peak_allocated_mib=new["peak_allocated_mib"])
        (RUN / "summary.json").write_text(json.dumps(summary, indent=2))
        archive_path = RUN.with_suffix(".zip")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(RUN.iterdir()):
                if path.is_file() and path.suffix in (".json", ".log", ".txt"):
                    archive.write(path, path.name)
        print(json.dumps(summary, indent=2))
        print("GỬI LẠI FILE NÀY:", archive_path)
        try:
            from google.colab import files
            files.download(str(archive_path))
        except Exception:
            print("Trong VS Code: dùng Colab Explorer để Download file ZIP ở đường dẫn trên.")
    ''')]
    cells.append(runtime_profile_cell())
    for index, entry in enumerate(cells):
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), f"cell_{index}", "exec")
    return dict(cells=cells, nbformat=4, nbformat_minor=5, metadata=dict(
        accelerator="GPU", colab=dict(name="colab_mer_attention_pilot.ipynb", provenance=[]),
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/colab_mer_attention_pilot.ipynb")
    parser.add_argument("--append-profile", action="store_true", help="Add/update only the final runtime profile cell; retain existing outputs")
    parser.add_argument("--append-factorized", action="store_true", help="Add/update only the factorized candidate cell; retain existing outputs")
    args = parser.parse_args()
    previous = None
    if args.output.exists():
        previous_bytes = args.output.read_bytes()
        previous = json.loads(previous_bytes.decode("utf-8"))
        if any(c.get("outputs") for c in previous.get("cells", [])):
            backup_dir = ROOT / "outputs/colab_attention_results/notebook_backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / (args.output.stem + "_" + hashlib.sha256(previous_bytes).hexdigest()[:16] + ".ipynb")
            if backup.exists() and backup.read_bytes() != previous_bytes:
                raise ValueError("Notebook backup identity mismatch")
            backup.write_bytes(previous_bytes)
            print("Previous outputs preserved:", backup)
    if args.append_profile or args.append_factorized:
        if previous is None:
            raise ValueError("Open notebook is required for --append-profile")
        entry = factorized_pilot_cell() if args.append_factorized else runtime_profile_cell()
        compile("".join(entry["source"]), "runtime_profile_cell", "exec")
        tag = "merit-factorized-pilot" if args.append_factorized else "merit-runtime-profile"
        found = [i for i, c in enumerate(previous["cells"]) if tag in c.get("metadata", {}).get("tags", [])]
        if found:
            previous["cells"][found[0]] = entry
        else:
            previous["cells"].append(entry)
        args.output.write_text(json.dumps(previous, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("Updated final runtime-profile cell; previous results retained:", args.output)
        return
    bundle, specs = build_payload()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    notebook = make_notebook(bundle, specs, commit)
    if previous is not None:
        for key in ("kernelspec", "language_info", "vscode"):
            if key in previous.get("metadata", {}):
                notebook["metadata"][key] = previous["metadata"][key]
    notebook["metadata"]["colab"]["name"] = args.output.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(args.output)
    print("Source archive SHA256:", hashlib.sha256(bundle).hexdigest())
    print("CT/GT download bytes:", sum(sum(s["file_sizes"].values()) for s in specs))


if __name__ == "__main__":
    main()
