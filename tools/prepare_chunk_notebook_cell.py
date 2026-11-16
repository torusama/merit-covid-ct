"""Append a standalone chunk-size diagnostic to the existing Colab notebook."""
import ast
import hashlib
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/colab_mer_attention_pilot.ipynb"
MARKER = "# MER_QUERY_CHUNK_EXPERIMENT_V1"


def prior_constants(book):
    for cell in reversed(book["cells"]):
        source = "".join(cell.get("source", []))
        if "FACTOR_BOOTSTRAP = " not in source or "FACTOR_SOURCES = " not in source:
            continue
        result = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in {"FACTOR_BOOTSTRAP", "FACTOR_SOURCES", "FACTOR_HASHES"}:
                    result[name] = ast.literal_eval(node.value)
        if set(result) == {"FACTOR_BOOTSTRAP", "FACTOR_SOURCES", "FACTOR_HASHES"}:
            return result
    raise ValueError("Existing factorized notebook cell was not found")


def main():
    book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if any(MARKER in "".join(cell.get("source", [])) for cell in book["cells"]):
        raise ValueError("Chunk experiment cell already exists; do not overwrite outputs")
    constants = prior_constants(book)
    sources = dict(constants["FACTOR_SOURCES"])
    for name in ("tools/merit_chunk_pilot.py", "tests/test_mer_chunk_equivalence.py"):
        sources[name] = (ROOT / name).read_text(encoding="utf-8")
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    preparation = "".join(book["cells"][2]["source"]) + "\n" + "".join(book["cells"][3]["source"])
    prefix = (MARKER + "\n" + "CHUNK_BOOTSTRAP = " + repr(constants["FACTOR_BOOTSTRAP"]) + "\n"
              + "CHUNK_PREPARATION = " + repr(preparation) + "\n"
              + "CHUNK_SOURCES = " + repr(sources) + "\nCHUNK_HASHES = " + repr(hashes) + "\n")
    body = '''
# Three native-512 TRAIN cases, same model/init/order; no accuracy claim.
if not all(name in globals() for name in ("PROJECT", "DATA", "RUN", "ENV", "BASE", "CASE_SPECS")):
    print("Initializing the existing frozen Colab snapshot.")
    exec(CHUNK_BOOTSTRAP, globals())
# Install only missing dependencies, then reuse or download and hash-check the
# exact three CT/mask pairs. This works after a Colab runtime reset as well.
PREPARATION_ERROR = None
exec(CHUNK_PREPARATION, globals())
if PREPARATION_ERROR:
    raise RuntimeError("Data or environment preparation failed: " + PREPARATION_ERROR)
import shutil
CHUNK_RUN = BASE / ("query_chunk_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
CHUNK_RUN.mkdir()
CHUNK_PROJECT = CHUNK_RUN / "source"
shutil.copytree(PROJECT, CHUNK_PROJECT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for name, source_text in CHUNK_SOURCES.items():
    if hashlib.sha256(source_text.encode()).hexdigest() != CHUNK_HASHES[name]:
        raise ValueError("Experiment code hash mismatch: " + name)
    path = CHUNK_PROJECT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source_text, encoding="utf-8")
CHUNK_ENV = dict(ENV, PYTHONPATH=os.pathsep.join([str(CHUNK_PROJECT / "src"), str(CHUNK_PROJECT)]))
(CHUNK_RUN / "provenance.json").write_text(json.dumps(dict(base_source_sha256=SOURCE_SHA256,
    experiment_source_sha256=CHUNK_HASHES, cases=[s["case_id"] for s in CASE_SPECS]), indent=2))
check = subprocess.run([sys.executable, "-m", "pytest", "tests/test_mer_chunk_equivalence.py", "-q",
    "--tb=short", "-p", "no:cacheprovider"], cwd=CHUNK_PROJECT, env=CHUNK_ENV,
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
(CHUNK_RUN / "equivalence.log").write_text(check.stdout, encoding="utf-8")
print(check.stdout)
import re
match = re.search(r"(\\d+) passed", check.stdout)
checks_passed = check.returncode == 0 and match is not None and int(match[1]) == 4 and "skipped" not in check.stdout
reports = {}
if checks_passed:
    for size in (128, 256, 512):
        name = "chunk_" + str(size)
        destination = CHUNK_RUN / (name + ".json")
        command = [sys.executable, "-u", "tools/merit_chunk_pilot.py", "--config",
            str(CHUNK_PROJECT / "configs/merit_native512.yaml"), "--data-dir", str(DATA),
            "--specs", str(RUN / "case-specifications.json"), "--output", str(destination),
            "--chunk", str(size), "--rounds", "1"]
        with (CHUNK_RUN / (name + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=CHUNK_PROJECT, env=CHUNK_ENV, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end="", flush=True)
            code = process.wait()
        report = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else {"status": "FAIL", "error": "Worker did not write report"}
        if code and report.get("status") == "PASS":
            report["status"] = "FAIL"
        report["exit_code"] = code
        reports[str(size)] = report
        print("Chunk", size, "status", report.get("status"), "mean seconds", report.get("mean_step_seconds"), flush=True)
base = reports.get("128", {})
summary = dict(status="PASS" if checks_passed and base.get("status") == "PASS" else "FAIL",
    correctness_tests_passed=checks_passed, cases=[s["case_id"] for s in CASE_SPECS],
    scope="3 TRAIN cases, native 512, one warmup and 6 measured updates per size; no validation or accuracy claim",
    metrics={})
for size, report in reports.items():
    same_init = (base.get("initial_state_sha256") is not None and
                 base.get("initial_state_sha256") == report.get("initial_state_sha256"))
    row = {k: report.get(k) for k in ("status", "error", "mean_step_seconds", "median_step_seconds",
        "measured_steps", "peak_allocated_mib", "peak_reserved_mib", "amp_fp32_retries")}
    row["same_initial_weights"] = same_init
    if report.get("status") == "PASS" and same_init and base.get("mean_step_seconds"):
        row["speed_ratio_vs_128"] = base["mean_step_seconds"] / report["mean_step_seconds"]
    summary["metrics"][size] = row
(CHUNK_RUN / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
result_zip = CHUNK_RUN.with_suffix(".zip")
with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in CHUNK_RUN.iterdir():
        if path.is_file() and path.suffix in (".json", ".log"):
            archive.write(path, path.name)
print("CHUNK EXPERIMENT SUMMARY", json.dumps(summary, indent=2), flush=True)
print("RESULT ZIP:", result_zip, flush=True)
try:
    from google.colab import files
    files.download(str(result_zip))
except Exception:
    print("Download the ZIP with Colab Explorer if needed.")
'''
    code = prefix + body
    compile(code, "colab_query_chunk_experiment", "exec")
    book["cells"].append(dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                              id=uuid.uuid4().hex[:8], source=code.splitlines(keepends=True)))
    NOTEBOOK.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Appended one diagnostic cell to", NOTEBOOK)


if __name__ == "__main__":
    main()
