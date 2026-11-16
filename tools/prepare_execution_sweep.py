"""Append a guarded native-MER execution sweep to the existing pilot notebook."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/colab_mer_attention_pilot.ipynb"
MARKER = "# MER_EXECUTION_SWEEP_V1"


def constant(book, name):
    for cell in reversed(book["cells"]):
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                    isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
                return ast.literal_eval(node.value)
    raise ValueError(name)


def main():
    book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if any(MARKER in "".join(c["source"]) for c in book["cells"]):
        raise ValueError("Sweep already exists; preserve its outputs")
    sources = dict(constant(book, "FACTOR_SOURCES"))
    sources["tools/merit_runtime_profile.py"] = constant(book, "PROFILE_SOURCE")
    for name in ("tools/merit_execution_tuning.py", "tests/test_mer_execution_tuning.py"):
        sources[name] = (ROOT / name).read_text(encoding="utf-8")
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}
    prefix = MARKER+"\n"+"SWEEP_BOOTSTRAP = "+repr(constant(book, "FACTOR_BOOTSTRAP"))+"\n"
    prefix += "SWEEP_PREPARATION = "+repr(constant(book, "CHUNK_PREPARATION"))+"\n"
    prefix += "SWEEP_SOURCES = "+repr(sources)+"\nSWEEP_HASHES = "+repr(hashes)+"\n"
    body = '''
if not all(n in globals() for n in ("PROJECT", "DATA", "RUN", "ENV", "BASE", "CASE_SPECS")):
    exec(SWEEP_BOOTSTRAP, globals())
PREPARATION_ERROR = None
exec(SWEEP_PREPARATION, globals())
if PREPARATION_ERROR:
    raise RuntimeError(PREPARATION_ERROR)
import shutil, statistics
SWEEP_RUN = BASE / ("execution_sweep_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
SWEEP_RUN.mkdir()
SWEEP_PROJECT = SWEEP_RUN / "source"
shutil.copytree(PROJECT, SWEEP_PROJECT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for name, text in SWEEP_SOURCES.items():
    assert hashlib.sha256(text.encode()).hexdigest() == SWEEP_HASHES[name], name
    destination = SWEEP_PROJECT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
SWEEP_ENV = dict(ENV, PYTHONPATH=os.pathsep.join([str(SWEEP_PROJECT / "src"), str(SWEEP_PROJECT)]))
(SWEEP_RUN / "provenance.json").write_text(json.dumps(dict(base_source_sha256=SOURCE_SHA256,
    experiment_source_sha256=SWEEP_HASHES, cases=CASE_SPECS, environment=ENVIRONMENT), indent=2))

def sweep_process(label, command):
    print("START", label, flush=True)
    with (SWEEP_RUN / (label + ".log")).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=SWEEP_PROJECT, env=SWEEP_ENV, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True); log.write(line); log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise

checks = sweep_process("compact-equivalence", [sys.executable, "-m", "pytest", "-q", "--tb=short",
    "-p", "no:cacheprovider", "tests/test_mer_execution_tuning.py"])
check_text = (SWEEP_RUN / "compact-equivalence.log").read_text()
checks_ok = checks == 0 and "14 passed" in check_text and "skipped" not in check_text
reports, guard_results = {}, {}
schedule = [("512_on_start",512,"on"),("1024_on",1024,"on"),("4096_on",4096,"on"),
            ("16384_on",16384,"on"),("16384_off",16384,"off"),("512_on_end",512,"on")]
if checks_ok:
    for label, chunk, recompute in schedule:
        guard = SWEEP_RUN / (label + "_guard.json")
        if label != "512_on_end":
            code = sweep_process(label + "_guard", [sys.executable, "-u", "tools/merit_execution_tuning.py",
                "--chunk", str(chunk), "--recompute", recompute, "--guard", "--output", str(guard)])
            guard_results[label] = json.loads(guard.read_text()) if guard.exists() else dict(status="FAIL", exit_code=code)
            if code or guard_results[label].get("status") != "PASS":
                reports[label] = dict(status="GUARD_FAIL")
                continue
        result = SWEEP_RUN / (label + ".json")
        code = sweep_process(label, [sys.executable, "-u", "tools/merit_execution_tuning.py", "--chunk",str(chunk),
            "--recompute",recompute,"--config",str(SWEEP_PROJECT / "configs/merit_native512.yaml"),
            "--data-dir",str(DATA),"--specs",str(RUN / "case-specifications.json"),"--output",str(result)])
        report = json.loads(result.read_text()) if result.exists() else dict(status="FAIL")
        report.update(chunk=chunk, recompute=recompute, exit_code=code)
        if code and report.get("status") == "PASS": report["status"] = "FAIL"
        reports[label] = report
        print("RESULT",label,report.get("status"),report.get("mean_step_seconds"),flush=True)

base = reports.get("512_on_start", {})
valid = {label:r for label,r in reports.items() if r.get("status")=="PASS" and
    r.get("initial_state_sha256") is not None and r.get("initial_state_sha256")==base.get("initial_state_sha256")}
best = min(valid, key=lambda label:valid[label]["mean_step_seconds"]) if valid else None
# Confirm the selected setting in another fresh process before profiling it.
if best:
    setting=valid[best]; result=SWEEP_RUN / "best-repeat.json"
    code=sweep_process("best-repeat",[sys.executable,"-u","tools/merit_execution_tuning.py","--chunk",str(setting["chunk"]),
        "--recompute",setting["recompute"],"--config",str(SWEEP_PROJECT/"configs/merit_native512.yaml"),
        "--data-dir",str(DATA),"--specs",str(RUN/"case-specifications.json"),"--output",str(result)])
    reports["best-repeat"]=json.loads(result.read_text()) if result.exists() else dict(status="FAIL")
    reports["best-repeat"].update(chunk=setting["chunk"],recompute=setting["recompute"])
    if code: reports["best-repeat"]["status"]="FAIL"
    profile_code = """import json
from unittest.mock import patch
from pathlib import Path
from tools.merit_execution_tuning import recomputation, execution, factor
from tools.merit_runtime_profile import run_profile
original=execution.configure_model_execution
def configure(model,settings):
 model=factor.enable_factorized_mer(original(model,settings));model.mer.query_chunk_size=CHUNK;return model
with patch.object(execution,'configure_model_execution',configure),recomputation(RECOMPUTE):
 report=run_profile(CONFIG,DATA,json.loads(Path(SPECS).read_text()),OUTPUT)
if report['status']!='PASS': raise SystemExit(1)
"""
    bindings=dict(CHUNK=setting["chunk"],RECOMPUTE=setting["recompute"]=="on",CONFIG=str(SWEEP_PROJECT/"configs/merit_native512.yaml"),
        DATA=str(DATA),SPECS=str(RUN/"case-specifications.json"),OUTPUT=str(SWEEP_RUN/"best-profile.json"))
    profile_code=''.join(key+'='+repr(value)+'\\n' for key,value in bindings.items())+profile_code
    sweep_process("best-profile",[sys.executable,"-u","-c",profile_code])

summary=dict(status="PASS" if checks_ok and best and reports.get("best-repeat",{}).get("status")=="PASS" else "FAIL",
    correctness_tests_passed=checks_ok, best_observed=best, native_guards=guard_results,
    scope="3 native-512 TRAIN cases, 6 measured updates per run; independent repeat and profile; no validation accuracy claim",
    metrics={})
for label,r in reports.items():
    summary["metrics"][label]={k:r.get(k) for k in ("status","error","chunk","recompute","mean_step_seconds",
        "median_step_seconds","peak_allocated_mib","peak_reserved_mib","amp_fp32_retries","initial_state_sha256")}
if "512_on_start" in valid and "512_on_end" in valid:
    summary["baseline_end_over_start"] = valid["512_on_end"]["mean_step_seconds"]/valid["512_on_start"]["mean_step_seconds"]
if best: summary["best_speed_ratio_vs_start"] = base["mean_step_seconds"]/valid[best]["mean_step_seconds"]
if best and reports.get("best-repeat",{}).get("status")=="PASS":
    repeat=reports["best-repeat"]
    summary["repeat_initial_weights_match"] = repeat.get("initial_state_sha256")==base.get("initial_state_sha256")
    summary["repeat_over_first"] = repeat["mean_step_seconds"]/valid[best]["mean_step_seconds"]
    summary["timing_stable_within_10_percent"] = (0.9 <= summary.get("baseline_end_over_start",0) <= 1.1 and
        0.9 <= summary["repeat_over_first"] <= 1.1)
    if not summary["repeat_initial_weights_match"]: summary["status"]="FAIL"
    elif not summary["timing_stable_within_10_percent"]: summary["status"]="INCONCLUSIVE"
(SWEEP_RUN/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
profile_path=SWEEP_RUN/"best-profile.json"
if profile_path.exists():
    profile=json.loads(profile_path.read_text())
    print("TUNED PROFILE",json.dumps({k:profile.get(k) for k in ("status","two_step_mean_seconds","stages","top_self_cpu","top_self_device")},indent=2))
result_zip=SWEEP_RUN.with_suffix(".zip")
with zipfile.ZipFile(result_zip,"w",zipfile.ZIP_DEFLATED) as archive:
    for path in SWEEP_RUN.iterdir():
        if path.is_file() and path.suffix in (".json",".log"): archive.write(path,path.name)
print("EXECUTION SWEEP SUMMARY",json.dumps(summary,indent=2),flush=True)
print("RESULT ZIP:",result_zip,flush=True)
try:
    from google.colab import files
    files.download(str(result_zip))
except Exception: print("Download result ZIP with Colab Explorer.")
'''
    source = prefix+body
    compile(source, "execution_sweep", "exec")
    book["cells"].append(dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
        id=uuid.uuid4().hex[:8], source=source.splitlines(keepends=True)))
    NOTEBOOK.write_text(json.dumps(book,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print("Appended execution sweep to", NOTEBOOK)


def parallel_sweep():
    book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    marker = "# MER_PARALLEL_REDUCTION_SWEEP_V1"
    if any(marker in "".join(c["source"]) for c in book["cells"]):
        raise ValueError("Parallel reduction sweep already exists")
    previous = next("".join(c["source"]) for c in reversed(book["cells"]) if
                    "".join(c["source"]).startswith(MARKER))
    tree = ast.parse(previous)
    start = next(n.lineno for n in tree.body if isinstance(n, ast.If))
    body = "".join(previous.splitlines(keepends=True)[start-1:])
    body = body.replace('"8 passed"', '"14 passed"').replace('"12 passed"', '"14 passed"')
    old = '''schedule = [("512_on_start",512,"on"),("1024_on",1024,"on"),("4096_on",4096,"on"),
            ("16384_on",16384,"on"),("16384_off",16384,"off"),("512_on_end",512,"on")]'''
    new = '''schedule = [("512_on_start",4096,"on"),("fast_4096_on",4096,"on"),
            ("fast_16384_on",16384,"on"),("fast_4096_off",4096,"off"),("512_on_end",4096,"on")]'''
    assert old in body
    body = body.replace(old, new)
    body = body.replace('str(guard)])', 'str(guard)] + (["--fast-reductions"] if label.startswith("fast_") else []))')
    body = body.replace('str(result)])', 'str(result)] + (["--fast-reductions"] if label.startswith("fast_") else []))', 1)
    body = body.replace('report.update(chunk=chunk, recompute=recompute, exit_code=code)',
        'report.update(chunk=chunk, recompute=recompute, fast_reductions=label.startswith("fast_"), exit_code=code)')
    body = body.replace('str(result)])', 'str(result)] + (["--fast-reductions"] if setting["fast_reductions"] else []))', 1)
    body = body.replace('reports["best-repeat"].update(chunk=setting["chunk"],recompute=setting["recompute"])',
        'reports["best-repeat"].update(chunk=setting["chunk"],recompute=setting["recompute"],fast_reductions=setting["fast_reductions"])')
    body = body.replace('from tools.merit_runtime_profile import run_profile',
        'from tools import merit_runtime_profile as profiler\noriginal_setup=profiler._setup\ndef setup(config):\n if FAST: config["training"]["deterministic"]=False\n return original_setup(config)\nprofiler._setup=setup\nrun_profile=profiler.run_profile')
    body = body.replace('bindings=dict(CHUNK=setting["chunk"],', 'bindings=dict(FAST=setting["fast_reductions"],CHUNK=setting["chunk"],')
    body = body.replace('"chunk","recompute","mean_step_seconds"', '"chunk","recompute","fast_reductions","mean_step_seconds"')
    body = body.replace('512_on_start', '4096_deterministic_start').replace('512_on_end', '4096_deterministic_end')
    body = body.replace('execution_sweep_', 'parallel_reduction_sweep_')
    body = body.replace('EXECUTION SWEEP SUMMARY', 'PARALLEL REDUCTION SWEEP SUMMARY')
    sources = dict(constant(book, "SWEEP_SOURCES"))
    for name in ("tools/merit_execution_tuning.py", "tests/test_mer_execution_tuning.py"):
        sources[name] = (ROOT / name).read_text(encoding="utf-8")
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()}
    prefix = marker+"\nSWEEP_BOOTSTRAP = "+repr(constant(book,"SWEEP_BOOTSTRAP"))+"\n"
    prefix += "SWEEP_PREPARATION = "+repr(constant(book,"SWEEP_PREPARATION"))+"\n"
    prefix += "SWEEP_SOURCES = "+repr(sources)+"\nSWEEP_HASHES = "+repr(hashes)+"\n"
    source = prefix+body
    compile(source,"parallel_reduction_sweep","exec")
    book["cells"].append(dict(cell_type="code",execution_count=None,metadata={},outputs=[],
        id=uuid.uuid4().hex[:8],source=source.splitlines(keepends=True)))
    NOTEBOOK.write_text(json.dumps(book,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print("Appended parallel-reduction comparison to", NOTEBOOK)


if __name__ == "__main__":
    parallel_sweep() if "--parallel" in sys.argv else main()
