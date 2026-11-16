"""Prepare a paired learning/held-out-volume diagnostic in the existing notebook."""
import hashlib
import json
from pathlib import Path
import uuid

from tools.prepare_execution_sweep import constant

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'notebooks/colab_mer_attention_pilot.ipynb'
    book=json.loads(path.read_text(encoding='utf-8'))
    marker='# MER_PAIRED_QUALITY_PILOT_V1'
    if any(marker in ''.join(c['source']) for c in book['cells']):raise ValueError('Already added; keep outputs')
    train=constant(book,'CASE_SPECS')
    split_raw=(ROOT/'splits/covid19_20_5fold.json').read_bytes().replace(b'\r\n',b'\n')
    split=json.loads(split_raw)
    validation=[]
    # Fixed case-ID order, chosen without consulting labels or model predictions.
    for case in sorted(split['folds'][0]['validation']['case_ids'])[:2]:
        spec=dict(case_id=case,group_id=case,ct_name=case+'_ct.nii.gz',gt_name=case+'_seg.nii.gz',sha256={})
        for kind,key in [('ct','ct_name'),('gt','gt_name')]:
            spec['sha256'][kind]=hashlib.sha256((ROOT/'data/covid19_20/Train'/spec[key]).read_bytes()).hexdigest()
        validation.append(spec)
    specs=dict(train=train,validation=validation,split_sha256_lf=hashlib.sha256(split_raw).hexdigest())
    sources=constant(book,'SWEEP_SOURCES')
    sources['tools/merit_quality_pilot.py']=(ROOT/'tools/merit_quality_pilot.py').read_text(encoding='utf-8')
    hashes={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()}
    prefix=marker+'\n'+''.join(name+' = '+repr(value)+'\n' for name,value in [
        ('QUALITY_BOOTSTRAP',constant(book,'SWEEP_BOOTSTRAP')),
        ('QUALITY_PREPARATION',constant(book,'SWEEP_PREPARATION')),
        ('QUALITY_SPECS',specs),('QUALITY_SOURCES',sources),('QUALITY_HASHES',hashes)])
    body='''
if not all(n in globals() for n in ("PROJECT","DATA","RUN","ENV","BASE","CASE_SPECS")):
    exec(QUALITY_BOOTSTRAP,globals())
# Use the existing hash-verified downloader for both TRAIN and validation cases.
saved_case_specs=CASE_SPECS
try:
    CASE_SPECS=QUALITY_SPECS["train"]+QUALITY_SPECS["validation"]
    PREPARATION_ERROR=None
    exec(QUALITY_PREPARATION,globals())
finally:
    CASE_SPECS=saved_case_specs
if PREPARATION_ERROR:raise RuntimeError(PREPARATION_ERROR)
import shutil, threading, time
QUALITY_RUN=BASE/("paired_quality_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
QUALITY_RUN.mkdir()
QUALITY_PROJECT=QUALITY_RUN/"source"
shutil.copytree(PROJECT,QUALITY_PROJECT,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
for name,text in QUALITY_SOURCES.items():
    assert hashlib.sha256(text.encode()).hexdigest()==QUALITY_HASHES[name]
    f=QUALITY_PROJECT/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text,encoding="utf-8")
QUALITY_ENV=dict(ENV,PYTHONPATH=os.pathsep.join([str(QUALITY_PROJECT/"src"),str(QUALITY_PROJECT)]))
spec_path=QUALITY_RUN/"specifications.json";spec_path.write_text(json.dumps(QUALITY_SPECS,indent=2))
(QUALITY_RUN/"provenance.json").write_text(json.dumps(dict(base_source_sha256=SOURCE_SHA256,
    source_sha256=QUALITY_HASHES,environment=ENVIRONMENT,selection="First two validation IDs alphabetically, independent of labels",
    initialization="Same seeded random weights; not the trained Kaggle checkpoint",pcg_policy="Active ramp for both diagnostic branches"),indent=2))
reports={}
for variant in ("reference","fast"):
    destination=QUALITY_RUN/(variant+".json")
    command=[sys.executable,"-u","tools/merit_quality_pilot.py","--config",str(QUALITY_PROJECT/"configs/merit_native512.yaml"),
        "--data-dir",str(DATA),"--specs",str(spec_path),"--output",str(destination),"--shared",str(QUALITY_RUN/"prediction-reference"),"--variant",variant]
    print("QUALITY START",variant,flush=True)
    with (QUALITY_RUN/(variant+".log")).open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,cwd=QUALITY_PROJECT,env=QUALITY_ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=1)
        monitor_stop=threading.Event()
        def monitor_quality(child=process,progress=destination.with_suffix('.progress.json'),stop=monitor_stop):
            while not stop.wait(30):
                if child.poll() is not None:return
                details={"pid":child.pid}
                try:
                    entry=json.loads(progress.read_text());details.update(entry)
                    details["phase_age_seconds"]=round(time.time()-entry["time"],1)
                except Exception:pass
                try:
                    gpu=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total","--format=csv,noheader"],
                        text=True,capture_output=True,timeout=5)
                    details["gpu_util_and_memory"]=gpu.stdout.strip()
                except Exception:pass
                try:
                    memory=Path('/proc/meminfo').read_text().splitlines()
                    details["host_ram"]=[x for x in memory if x.startswith(('MemAvailable:','SwapFree:'))]
                except Exception:pass
                print("QUALITY HEARTBEAT",json.dumps(details),flush=True)
                if details.get("phase") in ("load_batch","forward_loss","backward","optimizer") and details.get("phase_age_seconds",0)>180:
                    print("QUALITY STOP: one stage exceeded 180 seconds; retaining partial metrics and diagnostics.",flush=True)
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    return
        monitor=threading.Thread(target=monitor_quality,daemon=True);monitor.start()
        try:
            for line in process.stdout:print(line,end="",flush=True);log.write(line);log.flush()
            code=process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
        finally:
            monitor_stop.set();monitor.join(timeout=10)
    report=json.loads(destination.read_text()) if destination.exists() else dict(status="FAIL",error="Worker failed; inspect log",exit_code=code)
    if code or report.get("status")!="PASS":report["status"]="FAIL"
    reports[variant]=report
    if report.get("status")!="PASS":break
same=(len(reports)==2 and reports["reference"].get("initial_state_sha256") is not None and
      reports["reference"].get("initial_state_sha256")==reports["fast"].get("initial_state_sha256"))
summary=dict(execution_status="PASS" if same and all(r.get("status")=="PASS" for r in reports.values()) else "FAIL",
    scope="Learning-parity pilot from random weights, 24 updates per branch, 2 full held-out volumes; not final model accuracy",
    same_initial_weights=same,metrics={})
for label,r in reports.items():
    summary["metrics"][label]=dict(status=r.get("status"),validation=r.get("validation"),pcg_ramp=r.get("pcg_ramp"),
        first_loss=r.get("steps",[{}])[0].get("losses"),last_loss=r.get("steps",[{}])[-1].get("losses"),
        prediction_differences=r.get("prediction_differences"),source_files_unchanged=r.get("source_files_unchanged"))
if summary["execution_status"]=="PASS":
    a=reports["reference"]["validation"]["macro"];b=reports["fast"]["validation"]["macro"]
    summary["fast_minus_reference"]={key:b[key]["mean"]-a[key]["mean"] for key in ("dice","iou","recall","precision") if a[key]["mean"] is not None and b[key]["mean"] is not None}
    all_empty=all(row["prediction_foreground_voxels"]==0 for r in reports.values() for row in r["validation"]["per_case"])
    summary["quality_observation"]=("All predictions are background: segmentation quality is not established; equal zero Dice cannot validate useful-segmentation parity."
        if all_empty else "Diagnostic validation only: final trained-model quality still requires a trained checkpoint and complete validation.")
(QUALITY_RUN/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
result_zip=QUALITY_RUN.with_suffix(".zip")
with zipfile.ZipFile(result_zip,"w",zipfile.ZIP_DEFLATED) as archive:
    for f in QUALITY_RUN.iterdir():
        if f.is_file() and f.suffix in (".json",".log",".png"):archive.write(f,f.name)
print("PAIRED QUALITY SUMMARY",json.dumps(summary,indent=2),flush=True)
print("RESULT ZIP:",result_zip,flush=True)
try:
    from IPython.display import display,Image
    for f in sorted(QUALITY_RUN.glob("*.png")):display(Image(filename=str(f)))
    from google.colab import files
    files.download(str(result_zip))
except Exception:print("Download result ZIP with Colab Explorer.")
'''
    source=prefix+body;compile(source,'paired_quality','exec')
    book['cells'].append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],id=uuid.uuid4().hex[:8],source=source.splitlines(keepends=True)))
    path.write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Added paired quality cell; validation:',[s['case_id'] for s in validation])


if __name__=='__main__':main()
